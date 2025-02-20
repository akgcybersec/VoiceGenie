# VoiceGenie: Voice Assistant with Whisper and Ollama

VoiceGenie is a voice assistant that uses OpenAI's Whisper for speech recognition and Ollama for generating responses. The assistant can listen to your voice, transcribe it, and provide intelligent responses using a language model. It also supports text-to-speech functionality to speak the responses back to you.

## Features

- **Speech Recognition**: Uses OpenAI's Whisper model for accurate speech-to-text conversion.
- **Language Model**: Integrates with Ollama for generating intelligent responses.
- **Text-to-Speech**: Converts text responses to speech using gTTS (Google Text-to-Speech).
- **Conversation History**: Maintains a short history of the conversation for context-aware responses.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/voice-assistant.git
   cd voice-assistant
   pip install -r requirements.txt
   ```

2. **Set up Ollama:** 
   - Ensure Ollama is running and accessible at the specified URL (default: http://127.0.0.1:11434).
   - Make sure the desired model (default: llava) is available in Ollama.

3. **Run the assistant:**
   ```
   python agentx.py
   ```

## Usage
   - **Start the assistant**: Run the script, and the assistant will greet you.
   - **Speak to the assistant**: The assistant will listen for your voice and respond accordingly.
   - **Exit the assistant**: Say "goodbye", "exit", "quit", or "stop" to end the conversation.

## Configuration

You can configure the assistant using command-line arguments:

    --whisper-model: Choose the Whisper model size (tiny, base, small, medium, large).
    --ollama-url: Specify the URL of the Ollama API.
    --ollama-model: Specify the Ollama model to use.

Example:
```
python agentx.py --whisper-model small --ollama-url http://localhost:11434 --ollama-model mistral
```

## Requirements
```
sounddevice for audio recording
whisper for speech recognition
gTTS for text-to-speech
pygame for audio playback
aiohttp for HTTP requests
numpy for audio processing
```