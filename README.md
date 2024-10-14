# SpeechGenie

**SpeechGenie** is an application that allows users to ask questions through their microphone and receive audio responses. This project utilizes advanced speech processing models to seamlessly convert spoken questions into spoken answers.

## Features

- **Audio to Audio Q&A**: Speak your question, and receive an audio response.
- **Real-time Processing**: Fast transcription and response generation using cutting-edge models.
- **Simple Interface**: Speak, listen, and get your answers effortlessly.

## Technologies Used

- [Whisper](https://openai.com/blog/whisper) - for speech-to-text conversion.
- [Gradio](https://www.gradio.app) - for creating the user interface.
- [Llama 3](https://ai.facebook.com/blog/large-language-model-llama) - for generating answers from text-based questions.
- [Microsoft's Speech-to-Text](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/) - for converting text responses back into audio.

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/speechgenie.git
   ```
2. Navigate to the project directory:
   ```bash
   cd speechgenie
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Hugging Face API key:
   ```env
   HF_TOKEN=your_huggingface_api_key
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Once the app is running, click on the microphone button, speak your question, and wait for the audio response.

## How It Works

1. **Speech Recognition**: The app captures your audio question using the microphone and converts it into text using the Whisper model.
2. **Question Processing**: The transcribed question is then passed to the Llama 3 model, which generates a text-based response.
3. **Response Synthesis**: The text response is converted back into audio using Microsoft's speech-to-text model.
4. **Audio Playback**: The generated audio is played back to the user.

## API Keys and Configuration

- Make sure you have a Hugging Face API key. Add it to a `.env` file as follows:
   ```env
   HF_TOKEN=your_huggingface_api_key
   ```

## Contributing

Feel free to submit issues, feature requests, or pull requests. Contributions are always welcome!
