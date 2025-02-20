import os
import logging
import numpy as np
import sounddevice as sd
import whisper
import pygame
import tempfile
import asyncio
import aiohttp
import json
import argparse
import sys
import re
from gtts import gTTS
from typing import Optional, Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self, model_size: str = "tiny", ollama_api_url: str = "http://127.0.0.1:11434", 
                 ollama_model: str = "llava"):
        """Initialize the voice assistant with specified parameters."""
        logger.info("Initializing Voice Assistant...")
        
        # Initialize Whisper model
        logger.info(f"Loading Whisper model ({model_size})...")
        self.whisper_model = whisper.load_model(model_size)
        
        # Audio settings
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.silence_duration = 2.5
        self.chunk_duration = 0.1
        self.min_speech_duration = 0.5
        
        # Ollama API settings
        self.ollama_api_url = ollama_api_url
        self.ollama_model = ollama_model

         # Audio device settings
        self.device_retry_limit = 3
        self.device_retry_delay = 2  # seconds
        
        # Initialize pygame for audio playback
        pygame.init()
        pygame.mixer.init()
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 5

        # Add exit flag
        self.should_exit = False
        
        # Code-related request detection
        self.code_request_patterns = [
            r"write (a|the)? ?code",
            r"write (a|the)? ?program",
            r"create (a|the)? ?function",
            r"implement (a|the)?",
            r"code (a|the)?",
            r"program that",
            r"function to",
            r"class that",
            r"script for"
        ]
        
        # Patterns for detecting code blocks in response
        self.code_block_pattern = r"```[\w]*\n[\s\S]*?\n```"
        
        logger.info("Voice Assistant is ready!")

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama API is accessible and model is available."""
        url = f"{self.ollama_api_url}/api/tags"
        
        try:
            logger.info(f"Checking connection to Ollama at {self.ollama_api_url}...")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to connect to Ollama: HTTP {response.status}")
                        return False
                    
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    
                    if self.ollama_model not in models:
                        logger.warning(f"Model '{self.ollama_model}' not found in Ollama. Available models: {models}")
                        return False
                    
                    logger.info(f"Successfully connected to Ollama. Model '{self.ollama_model}' is available.")
                    return True
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama connection: {e}")
            return False

    async def record_audio(self) -> Tuple[Optional[np.ndarray], int]:
        """Record audio until silence is detected, with improved error handling."""
        logger.info("Listening for speech...")
        
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        silence_chunks = int(self.silence_duration / self.chunk_duration)
        min_speech_chunks = int(self.min_speech_duration / self.chunk_duration)
        
        audio_chunks = []
        silence_count = 0
        speech_detected = False
        total_chunks = 0
        
        try:
            # Get list of available devices
            devices = sd.query_devices()
            input_device = None
            
            # Find first working input device
            for device in devices:
                if device['max_input_channels'] > 0:
                    try:
                        sd.check_input_settings(
                            device=device['name'],
                            channels=1,
                            dtype=np.float32,
                            samplerate=self.sample_rate
                        )
                        input_device = device['name']
                        break
                    except sd.PortAudioError:
                        continue
            
            if input_device is None:
                logger.error("No working input device found")
                return None, self.sample_rate
            
            # Create stream with explicit device
            stream = sd.InputStream(
                device=input_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_samples,
                latency='high',  # Use high latency for better stability
            )
            
            # Test stream before recording
            with stream:
                stream.read(chunk_samples)  # Test read
                await asyncio.sleep(0.1)    # Short pause
                
                while True:
                    try:
                        audio_chunk, _ = stream.read(chunk_samples)
                        audio_chunks.append(audio_chunk)
                        
                        # Calculate RMS amplitude
                        amplitude = np.sqrt(np.mean(audio_chunk**2))
                        
                        if amplitude > self.silence_threshold:
                            silence_count = 0
                            speech_detected = True
                        else:
                            silence_count += 1
                        
                        total_chunks += 1
                        
                        # Check stop conditions
                        if speech_detected:
                            if silence_count >= silence_chunks and total_chunks >= min_speech_chunks:
                                break
                        elif total_chunks > silence_chunks * 2:
                            return None, self.sample_rate
                        
                        if total_chunks > int(30 / self.chunk_duration):  # 30 seconds max
                            break
                        
                        await asyncio.sleep(0.1)
                        
                    except (sd.PortAudioError, OSError) as e:
                        logger.error(f"Stream read error: {e}")
                        return None, self.sample_rate
            
            if not speech_detected or len(audio_chunks) < min_speech_chunks:
                logger.info("No speech detected")
                return None, self.sample_rate
            
            # Combine all audio chunks
            audio_data = np.concatenate(audio_chunks)
            logger.info(f"Recorded {len(audio_data) / self.sample_rate:.1f} seconds of audio")
            
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            # Add small delay before retry
            await asyncio.sleep(1)
            return None, self.sample_rate

    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper with additional filtering."""
        if audio_data is None:
            return None
                
        logger.info("Transcribing audio...")
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                import wave
                with wave.open(fp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
                
                # Transcribe audio
                result = self.whisper_model.transcribe(
                    fp.name,
                    fp16=False,
                    language="en",
                    beam_size=3,
                    temperature=0.0,
                )
                
                os.unlink(fp.name)  # Clean up temporary file
                
                text = result["text"].strip()
                
                # Add filtering logic
                if not text or len(text) < 3:
                    logger.info("Transcription too short, ignoring")
                    return None
                    
                if is_repetitive_speech(text):
                    logger.info("Detected repetitive speech pattern, ignoring")
                    return None
                    
                logger.info(f"Transcribed text: {text}")
                return text
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def is_code_request(self, text: str) -> bool:
        """Determine if the user is requesting code."""
        text = text.lower()
        for pattern in self.code_request_patterns:
            if re.search(pattern, text):
                return True
        return False

    def extract_readable_content(self, response: str) -> List[str]:
        """Extract readable content, removing code blocks."""
        # Replace all code blocks with placeholders
        readable_segments = []
        
        # Check if the response contains code blocks
        has_code = bool(re.search(self.code_block_pattern, response))
        
        if has_code:
            # Split by code blocks
            pieces = re.split(self.code_block_pattern, response)
            
            # Notify about code
            readable_segments.append("I've generated code in my response. I'll read just the explanation.")
            
            # Add non-code pieces
            for piece in pieces:
                if piece.strip():
                    readable_segments.append(piece.strip())
        else:
            # No code blocks, return the original response
            readable_segments.append(response)
            
        return readable_segments

    async def speak_text(self, text: str) -> None:
        """Convert text to speech and play it."""
        if not text:
            return
            
        logger.info(f"Speaking: {text}")
        try:
            # Split text into smaller chunks
            chunks = self._split_text_into_chunks(text)
            
            for chunk in chunks:
                tts = gTTS(text=chunk, lang="en", slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    pygame.mixer.music.load(fp.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                        
                    os.unlink(fp.name)
                    
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")

    def _split_text_into_chunks(self, text: str, max_chars: int = 500) -> list:
        """Split long text into smaller chunks for TTS."""
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        for sentence in text.split(". "):
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    async def get_ai_response(self, user_input: str) -> Optional[Dict]:
        """Generate a single streaming response using Ollama."""
        url = f"{self.ollama_api_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        
        conversation_context = "\n".join(
            [f"Human: {entry['user']}\nAssistant: {entry['assistant']}" 
             for entry in self.conversation_history]
        )
        
        prompt = f"{conversation_context}\nHuman: {user_input}\nAssistant:"
        
        # Check if this is a code request
        is_code_request = self.is_code_request(user_input)
        if is_code_request:
            logger.info("Detected code request. Will limit spoken response.")
        
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
        }
        
        try:
            full_response = ""
            current_sentence = ""
            current_speech_task = None  # Track current speech task
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if line:
                            try:
                                json_response = json.loads(line)
                                text = json_response.get("response", "")
                                if text:
                                    print(text, end="", flush=True)
                                    full_response += text
                                    
                                    # For code requests, we'll gather the full response first
                                    if not is_code_request:
                                        current_sentence += text
                                        
                                        # Check for sentence completion
                                        if text.endswith(('.', '!', '?')) and len(current_sentence) > 10:
                                            # Only speak if there's no active speech task
                                            sentence_to_speak = current_sentence.strip()
                                            if current_speech_task:
                                                await current_speech_task
                                            current_speech_task = asyncio.create_task(
                                                self.speak_text(sentence_to_speak)
                                            )
                                            current_sentence = ""
                                        
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n")
                    
                    # Handle remaining text after stream ends
                    if current_speech_task:
                        await current_speech_task
                    
                    if is_code_request:
                        # For code requests, speak the readable segments once at the end
                        readable_segments = self.extract_readable_content(full_response)
                        for segment in readable_segments:
                            await self.speak_text(segment)
                    elif current_sentence.strip():
                        # Speak any remaining non-code text
                        await self.speak_text(current_sentence.strip())
                    
                    return {
                        "text": full_response,
                        "is_code_request": is_code_request
                    }
                    
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None

    def _split_text_into_chunks(self, text: str, max_chars: int = 500) -> list:
        """Split long text into smaller chunks for TTS, avoiding mid-sentence splits."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split on sentence endings
        sentences = re.split('([.!?]+)', text)
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + ending
            
            if len(current_chunk) + len(full_sentence) <= max_chars:
                current_chunk += full_sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    async def conversation_loop(self) -> None:
        """Main conversation loop with improved exit command handling."""
        logger.info("Starting conversation loop...")
        
        # Check Ollama connection before starting
        if not await self.check_ollama_connection():
            logger.error(f"Cannot connect to Ollama at {self.ollama_api_url} or model '{self.ollama_model}' is not available.")
            await self.speak_text(f"Error connecting to Ollama. Please ensure Ollama is running and the model {self.ollama_model} is available.")
            return
        
        # Initial greeting
        await self.speak_text("Voice assistant is ready. How can I help you?")
        
        while not self.should_exit:
            try:
                # Record and transcribe audio
                audio_data, _ = await self.record_audio()
                if audio_data is None:
                    continue
                    
                user_input = await self.transcribe_audio(audio_data)
                if not user_input:
                    continue
                
                # Print what was recognized
                print(f"\nYou: {user_input}")
                
                # Check for exit commands
                exit_phrases = ["bye", "goodbye", "exit", "quit", "stop"]
                if any(phrase in user_input.lower().strip() for phrase in exit_phrases):
                    await self.speak_text("Goodbye! Have a great day!")
                    logger.info("Exit command received. Shutting down...")
                    self.should_exit = True
                    break
                
                # Get and speak AI response
                response_data = await self.get_ai_response(user_input)
                if response_data:
                    response_text = response_data["text"]
                    
                    # Update conversation history
                    self.conversation_history.append({
                        "user": user_input,
                        "assistant": response_text
                    })
                    if len(self.conversation_history) > self.max_history_length:
                        self.conversation_history.pop(0)
                        
            except asyncio.CancelledError:
                logger.info("Conversation loop cancelled. Cleaning up...")
                await self.speak_text("Goodbye! Have a great day!")
                self.should_exit = True
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                if not self.should_exit:  # Only sleep if we're not trying to exit
                    await asyncio.sleep(1)

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        try:
            pygame.mixer.music.stop()  # Stop any playing audio
            pygame.mixer.quit()
            pygame.quit()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def is_repetitive_speech(text: str, threshold: float = 0.8) -> bool:
    """
    Check if the transcribed text contains excessive repetition.
    
    Args:
        text: The transcribed text to analyze
        threshold: Maximum allowed repetition ratio (0.0 to 1.0)
    
    Returns:
        bool: True if text is overly repetitive, False otherwise
    """
    if not text:
        return False
        
    # Split into words and remove empty strings
    words = [w.strip().lower() for w in text.split() if w.strip()]
    
    if len(words) < 4:  # Too short to analyze
        return False
        
    # Count unique words
    unique_words = set(words)
    
    # Calculate repetition ratio
    repetition_ratio = 1 - (len(unique_words) / len(words))
    
    return repetition_ratio > threshold


async def main():
    """Main function with improved signal handling."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant using Whisper and Ollama LLMs",
        epilog="Note: This program requires Ollama to be running. Visit https://ollama.ai for setup instructions."
    )
    
    parser.add_argument(
        "--whisper-model", 
        default="tiny", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for speech recognition (default: tiny)"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="URL of the Ollama API (default: http://127.0.0.1:11434)"
    )
    
    parser.add_argument(
        "--ollama-model",
        default="llava",
        help="Ollama model to use for responses (default: llava)"
    )
    
    args = parser.parse_args()
    
    # Create and start the voice assistant
    assistant = VoiceAssistant(
        model_size=args.whisper_model,
        ollama_api_url=args.ollama_url,
        ollama_model=args.ollama_model
    )

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(assistant, loop)))
    
    try:
        await assistant.conversation_loop()
    finally:
        assistant.cleanup()

async def shutdown(assistant: VoiceAssistant, loop: asyncio.AbstractEventLoop):
    """Graceful shutdown handler."""
    logger.info("Shutdown initiated...")
    assistant.should_exit = True
    try:
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        loop.stop()
    
    try:
        await assistant.conversation_loop()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    # Add signal import
    import signal
    asyncio.run(main())