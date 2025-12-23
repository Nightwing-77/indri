import io
import time
import json
import uuid
import torch
import random
import asyncio
import traceback
import torchaudio
from pathlib import Path
from typing import AsyncGenerator
from queue import Queue
from threading import Thread
from torio.io import CodecConfig

from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.tts import TTS
from src.models import (
    TTSSpeakersResponse, Speakers, TTSRequest,
    SpeakerTextRequest, SpeakerTextResponse, AudioFeedbackRequest,
    AudioOutput, TTSMetrics
)
from src.logger import get_logger
from src.models import SPEAKER_MAP
from src.launcher import _add_shutdown_handlers
from src.db.feedback import RealFakeFeedbackDB

logger = get_logger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://indri-ui.vercel.app",
        "https://indrivoice.ai",
        "https://www.indrivoice.ai",
        "https://indrivoice.io",
        "https://www.indrivoice.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-sample-id", "x-request-id", "x-metrics"]
)


class StreamBuffer:
    """Ultra-aggressive word-level streaming buffer"""
    def __init__(self, min_words=3, max_words=15):
        self.buffer = []
        self.min_words = min_words
        self.max_words = max_words
        
    def add_word(self, word: str) -> str | None:
        """Add word and return chunk when ready"""
        if not word.strip():
            return None
            
        self.buffer.append(word)
        
        # Check for sentence end or max words
        has_ending = any(word.rstrip().endswith(p) for p in ['.', '!', '?', ';', ':'])
        
        if has_ending and len(self.buffer) >= self.min_words:
            chunk = ' '.join(self.buffer)
            self.buffer = []
            return chunk
        elif len(self.buffer) >= self.max_words:
            chunk = ' '.join(self.buffer)
            self.buffer = []
            return chunk
            
        return None
    
    def flush(self) -> str | None:
        """Get remaining words"""
        if self.buffer:
            chunk = ' '.join(self.buffer)
            self.buffer = []
            return chunk
        return None


class ParallelTTSProcessor:
    """Parallel processing queue for zero-gap streaming"""
    def __init__(self, tts_model, max_workers=2):
        self.tts_model = tts_model
        self.max_workers = max_workers
        
    async def process_chunks_parallel(self, text_chunks: list[str], speaker: str, request_id: str):
        """Process multiple chunks in parallel"""
        tasks = []
        for i, chunk in enumerate(text_chunks):
            task = self._generate_chunk(chunk, speaker, f"{request_id}-{i}")
            tasks.append(task)
        
        # Process with controlled parallelism
        results = []
        for i in range(0, len(tasks), self.max_workers):
            batch = tasks[i:i+self.max_workers]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def _generate_chunk(self, text: str, speaker: str, chunk_id: str):
        """Generate single audio chunk"""
        try:
            results: AudioOutput = await self.tts_model.generate_async(
                text=text,
                speaker=speaker,
                request_id=chunk_id
            )
            
            audio_tensor = torch.from_numpy(results.audio)
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate=results.sample_rate,
                format='mp3',
                encoding='PCM_S',
                bits_per_sample=16,
                backend='ffmpeg',
                compression=CodecConfig(bit_rate=64000)
            )
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Chunk generation failed: {e}")
            return None


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    FAST streaming WebSocket - word by word processing
    
    Send: {"text": "word"} or {"text": "word", "speaker": "alice"}
    Send: {"done": true} when finished
    
    Receive: {"audio": "<base64>", "id": 0}
    """
    await websocket.accept()
    
    request_id = str(uuid.uuid4())
    speaker = "🇮🇳 👨 politician"  # Default speaker - use actual value not enum
    buffer = StreamBuffer(min_words=3, max_words=10)
    chunk_id = 0
    
    # Task queue for parallel processing
    processing_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    async def audio_generator():
        """Background task to generate audio"""
        nonlocal chunk_id
        while True:
            try:
                item = await processing_queue.get()
                if item is None:  # Poison pill
                    break
                
                text_chunk = item
                logger.info(f"[{request_id}] Generating chunk {chunk_id}: {text_chunk[:50]}...")
                
                # Use same lookup as /tts endpoint
                speaker_id = SPEAKER_MAP.get(speaker, {'id': None}).get('id')
                
                if speaker_id is None:
                    logger.error(f"[{request_id}] Invalid speaker: {speaker}")
                    await output_queue.put({
                        'error': f'Invalid speaker: {speaker}',
                        'error_type': 'invalid_speaker',
                        'available_speakers': [str(s) for s in SPEAKER_MAP.keys()]
                    })
                    continue
                
                try:
                    results: AudioOutput = await tts_model.generate_async(
                        text=text_chunk,
                        speaker=speaker_id,
                        request_id=f"{request_id}-{chunk_id}"
                    )
                except Exception as model_error:
                    logger.error(f"[{request_id}] Model generation error: {model_error}\n{traceback.format_exc()}")
                    await output_queue.put({
                        'error': f'Model generation failed: {str(model_error)}',
                        'error_type': 'model_error',
                        'chunk_text': text_chunk,
                        'traceback': traceback.format_exc()
                    })
                    continue
                
                try:
                    audio_tensor = torch.from_numpy(results.audio)
                    buffer_io = io.BytesIO()
                    torchaudio.save(
                        buffer_io,
                        audio_tensor,
                        sample_rate=results.sample_rate,
                        format='mp3',
                        encoding='PCM_S',
                        bits_per_sample=16,
                        backend='ffmpeg',
                        compression=CodecConfig(bit_rate=64000)
                    )
                    buffer_io.seek(0)
                except Exception as encode_error:
                    logger.error(f"[{request_id}] Audio encoding error: {encode_error}\n{traceback.format_exc()}")
                    await output_queue.put({
                        'error': f'Audio encoding failed: {str(encode_error)}',
                        'error_type': 'encoding_error',
                        'chunk_text': text_chunk,
                        'traceback': traceback.format_exc()
                    })
                    continue
                
                logger.info(f"[{request_id}] Generated chunk {chunk_id} successfully")
                
                await output_queue.put({
                    'id': chunk_id,
                    'audio': buffer_io.getvalue(),
                    'text': text_chunk
                })
                chunk_id += 1
                
            except Exception as e:
                logger.error(f"[{request_id}] Generator error: {e}\n{traceback.format_exc()}")
                await output_queue.put({
                    'error': f'Unexpected error in generator: {str(e)}',
                    'error_type': 'generator_error',
                    'traceback': traceback.format_exc()
                })
    
    async def audio_sender():
        """Background task to send audio as soon as ready"""
        while True:
            try:
                item = await output_queue.get()
                if item is None:  # Poison pill
                    break
                
                if 'error' in item:
                    logger.error(f"[{request_id}] Sending error to client: {item}")
                    await websocket.send_json(item)
                    continue
                
                import base64
                audio_b64 = base64.b64encode(item['audio']).decode('utf-8')
                
                await websocket.send_json({
                    'audio': audio_b64,
                    'id': item['id'],
                    'text': item['text']
                })
                
            except WebSocketDisconnect:
                logger.info(f"[{request_id}] WebSocket disconnected in sender")
                break
            except Exception as e:
                logger.error(f"[{request_id}] Sender error: {e}\n{traceback.format_exc()}")
                try:
                    await websocket.send_json({
                        'error': f'Sender error: {str(e)}',
                        'error_type': 'sender_error',
                        'traceback': traceback.format_exc()
                    })
                except:
                    pass
                break
    
    # Start background workers
    generator_task = asyncio.create_task(audio_generator())
    sender_task = asyncio.create_task(audio_sender())
    
    try:
        logger.info(f"[{request_id}] WebSocket connected, default speaker: {speaker}")
        
        while True:
            data = await websocket.receive_json()
            
            if data.get('done'):
                logger.info(f"[{request_id}] Received done signal, flushing buffer...")
                # Flush remaining
                remaining = buffer.flush()
                if remaining:
                    logger.info(f"[{request_id}] Flushing remaining text: {remaining}")
                    await processing_queue.put(remaining)
                
                # Stop workers
                await processing_queue.put(None)
                await generator_task
                await output_queue.put(None)
                await sender_task
                
                logger.info(f"[{request_id}] Streaming complete, {chunk_id} chunks generated")
                await websocket.send_json({'done': True, 'total': chunk_id})
                break
            
            if 'speaker' in data:
                # Accept speaker exactly like /tts endpoint does
                speaker = data['speaker']
                logger.info(f"[{request_id}] Speaker set to: {speaker}")
            
            if 'text' in data:
                word = data['text']
                chunk = buffer.add_word(word)
                
                if chunk:
                    logger.info(f"[{request_id}] Buffer full, queueing chunk: {chunk[:50]}...")
                    # Queue for processing immediately
                    await processing_queue.put(chunk)
    
    except WebSocketDisconnect:
        logger.info(f"[{request_id}] WebSocket disconnected")
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] Invalid JSON received: {e}")
        try:
            await websocket.send_json({
                'error': f'Invalid JSON: {str(e)}',
                'error_type': 'json_error'
            })
        except:
            pass
    except Exception as e:
        logger.error(f"[{request_id}] WebSocket error: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({
                'error': str(e),
                'error_type': 'websocket_error',
                'traceback': traceback.format_exc()
            })
        except:
            pass
    finally:
        # Cleanup
        logger.info(f"[{request_id}] Cleaning up WebSocket connection")
        try:
            await processing_queue.put(None)
            await output_queue.put(None)
        except:
            pass


@app.post("/stream")
async def stream_tts(request: TTSRequest):
    """
    HTTP streaming endpoint - outputs audio chunks as they're ready
    Returns newline-delimited JSON (NDJSON)
    """
    request_id = str(uuid.uuid4())
    
    async def generate():
        try:
            speaker_id = SPEAKER_MAP.get(request.speaker, {'id': None}).get('id')
            if speaker_id is None:
                logger.error(f"[{request_id}] Invalid speaker: {request.speaker}")
                yield json.dumps({
                    'error': f'Invalid speaker: {request.speaker}',
                    'error_type': 'invalid_speaker',
                    'available_speakers': [str(s) for s in SPEAKER_MAP.keys()]
                }) + '\n'
                return
            
            logger.info(f"[{request_id}] Starting HTTP stream, speaker: {request.speaker}")
            
            # Split into words
            words = request.text.split()
            buffer = StreamBuffer(min_words=3, max_words=10)
            chunk_id = 0
            
            for word in words:
                chunk = buffer.add_word(word)
                
                if chunk:
                    logger.info(f"[{request_id}] Stream chunk {chunk_id}: {chunk[:50]}...")
                    
                    try:
                        # Generate audio
                        results: AudioOutput = await tts_model.generate_async(
                            text=chunk,
                            speaker=speaker_id,
                            request_id=f"{request_id}-{chunk_id}"
                        )
                    except Exception as model_error:
                        logger.error(f"[{request_id}] Model generation error: {model_error}\n{traceback.format_exc()}")
                        yield json.dumps({
                            'error': f'Model generation failed: {str(model_error)}',
                            'error_type': 'model_error',
                            'chunk_text': chunk,
                            'chunk_id': chunk_id,
                            'traceback': traceback.format_exc()
                        }) + '\n'
                        continue
                    
                    try:
                        audio_tensor = torch.from_numpy(results.audio)
                        buffer_io = io.BytesIO()
                        torchaudio.save(
                            buffer_io,
                            audio_tensor,
                            sample_rate=results.sample_rate,
                            format='mp3',
                            encoding='PCM_S',
                            bits_per_sample=16,
                            backend='ffmpeg',
                            compression=CodecConfig(bit_rate=64000)
                        )
                        buffer_io.seek(0)
                    except Exception as encode_error:
                        logger.error(f"[{request_id}] Audio encoding error: {encode_error}\n{traceback.format_exc()}")
                        yield json.dumps({
                            'error': f'Audio encoding failed: {str(encode_error)}',
                            'error_type': 'encoding_error',
                            'chunk_text': chunk,
                            'chunk_id': chunk_id,
                            'traceback': traceback.format_exc()
                        }) + '\n'
                        continue
                    
                    import base64
                    audio_b64 = base64.b64encode(buffer_io.getvalue()).decode('utf-8')
                    
                    yield json.dumps({
                        'id': chunk_id,
                        'audio': audio_b64,
                        'text': chunk,
                        'sample_rate': results.sample_rate
                    }) + '\n'
                    
                    chunk_id += 1
            
            # Flush remaining
            remaining = buffer.flush()
            if remaining:
                logger.info(f"[{request_id}] Stream final chunk: {remaining}")
                
                try:
                    results: AudioOutput = await tts_model.generate_async(
                        text=remaining,
                        speaker=speaker_id,
                        request_id=f"{request_id}-final"
                    )
                    
                    audio_tensor = torch.from_numpy(results.audio)
                    buffer_io = io.BytesIO()
                    torchaudio.save(
                        buffer_io,
                        audio_tensor,
                        sample_rate=results.sample_rate,
                        format='mp3',
                        encoding='PCM_S',
                        bits_per_sample=16,
                        backend='ffmpeg',
                        compression=CodecConfig(bit_rate=64000)
                    )
                    buffer_io.seek(0)
                    
                    import base64
                    audio_b64 = base64.b64encode(buffer_io.getvalue()).decode('utf-8')
                    
                    yield json.dumps({
                        'id': chunk_id,
                        'audio': audio_b64,
                        'text': remaining,
                        'sample_rate': results.sample_rate
                    }) + '\n'
                except Exception as final_error:
                    logger.error(f"[{request_id}] Final chunk error: {final_error}\n{traceback.format_exc()}")
                    yield json.dumps({
                        'error': f'Final chunk failed: {str(final_error)}',
                        'error_type': 'final_chunk_error',
                        'chunk_text': remaining,
                        'traceback': traceback.format_exc()
                    }) + '\n'
            
            logger.info(f"[{request_id}] HTTP stream complete, {chunk_id} chunks generated")
            yield json.dumps({'done': True, 'total_chunks': chunk_id}) + '\n'
            
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}\n{traceback.format_exc()}")
            yield json.dumps({
                'error': str(e),
                'error_type': 'stream_error',
                'traceback': traceback.format_exc()
            }) + '\n'
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-request-id": request_id
        }
    )


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received text: {request.text} with speaker: {request.speaker}', extra={'request_id': request_id})

    try:
        speaker = SPEAKER_MAP.get(request.speaker, {'id': None}).get('id')

        if speaker is None:
            raise HTTPException(status_code=400, detail=f'Speaker {speaker} not supported')

        results: AudioOutput = await tts_model.generate_async(
            text=request.text,
            speaker=speaker,
            request_id=request_id
        )
        metrics: TTSMetrics = results.audio_metrics

        audio_tensor = torch.from_numpy(results.audio)
        logger.info(f'Audio shape: {audio_tensor.shape}', extra={'request_id': request_id})

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate=results.sample_rate,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
        buffer.seek(0)

    except Exception as e:
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}", extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(request_id) + ' ' + str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech_completion.wav",
        "x-request-id": request_id,
        "x-metrics": json.dumps(metrics.model_dump())
    }

    return Response(
        content=buffer.getvalue(),
        headers=headers,
        media_type="audio/wav"
    )


@app.post("/audio_completion")
async def audio_completion(text: str, file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received text: {text}', extra={'request_id': request_id})

    try:
        allowed_types = {'.wav', '.mp3', '.m4a'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f'Unsupported file type. Allowed types: {", ".join(allowed_types)}'
            )

        contents = await file.read()
        logger.info(f'Received audio file: {file.filename}', extra={'request_id': request_id})
        audio, sr = torchaudio.load(io.BytesIO(contents))

        results: AudioOutput = await tts_model.generate_async(
            text=text,
            speaker='[spkr_unk]',
            audio=audio,
            sample_rate=sr,
            request_id=request_id
        )
        metrics: TTSMetrics = results.audio_metrics

        audio_tensor = torch.from_numpy(results.audio)
        logger.info(f'Audio shape: {audio_tensor.shape}', extra={'request_id': request_id})

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate=results.sample_rate,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
        buffer.seek(0)

    except Exception as e:
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}", extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(request_id) + ' ' + str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech_completion.wav",
        "x-request-id": request_id,
        "x-metrics": json.dumps(metrics.model_dump())
    }

    return Response(
        content=buffer.getvalue(),
        headers=headers,
        media_type="audio/wav"
    )


@app.get("/speakers", response_model=TTSSpeakersResponse)
async def available_speakers():
    return {
        "speakers": [s for s in Speakers]
    }


@app.post("/speaker_text", response_model=SpeakerTextResponse)
async def speaker_text(request: SpeakerTextRequest):
    speaker_text = SPEAKER_MAP.get(request.speaker, {'text': None}).get('text')

    if speaker_text is None:
        raise HTTPException(status_code=400, detail=f'Speaker {request.speaker} not supported')

    return {
        "speaker_text": random.choice(speaker_text)
    }


@app.get("/sample_audio")
async def sample_audio():
    try:
        choice = random.choice(sample_audio_files)
        logger.info(f'Serving sample audio: {choice}')

        aud, sr = torchaudio.load(f'sample/{choice}.wav')

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            aud,
            sample_rate=sr,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
        buffer.seek(0)

        headers = {
            "Content-Type": "audio/wav",
            "Content-Disposition": "attachment; filename=speech.wav",
            "x-sample-id": choice
        }

        return Response(
            content=buffer.getvalue(),
            headers=headers,
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(f'Error in sampling audio: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio_feedback")
async def audio_feedback(request: AudioFeedbackRequest):
    try:
        assert request.id in sample_audio_files, f'Sample audio with id {request.id} not found'
        assert request.feedback in [-1, 1], f'Feedback must be -1 or 1'
    except Exception as e:
        logger.error(f'Error in audio feedback: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f'Received audio feedback for {request.id}: {request.feedback}')
    RealFakeFeedbackDB().insert_feedback(request.id, request.feedback)

    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='11mlabs/indri-0.1-124m-tts', help='HF model repository id')
    parser.add_argument('--device', type=str, default='cuda:0', required=False, help='Device to use for inference')
    parser.add_argument('--port', type=int, default=8000, required=False, help='Port to run the server on')

    args = parser.parse_args()

    logger.info(f'Loading model from {args.model_path} on {args.device} and starting server on port {args.port}')

    global tts_model
    tts_model = TTS(model_path=args.model_path, device=args.device)

    file_names = list(Path('sample/').resolve().glob('**/*.wav'))
    logger.info(f'Found {len(file_names)} sample audio files')

    global sample_audio_files
    sample_audio_files = [f.stem for f in file_names]

    server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=args.port))
    _add_shutdown_handlers(app, server)

    server.run()
