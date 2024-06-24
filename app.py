from warnings import filterwarnings
filterwarnings("ignore")
from transformers import pipeline
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import gradio as gr
from huggingface_hub import HfFolder
import requests
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torchaudio
import os 


#below is the transcriber pipeline that loads whisper model
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small.en", device=device
)

#convert audio in to text
def transcribe(audio):
  print("Listening your query")
  result = transcriber(audio)
  return result['text']

#uses hosted api of Llama-3 model gives response
def query(text, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    hf_folder = HfFolder()
    headers = {"Authorization": f"Bearer {hf_folder.get_token()}"}
    payload = {"inputs": text}

    print(f"Querying...: {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    print(response.json()[0]['generated_text'][len(text) + 1 :])
    return response.json()[0]['generated_text'][len(text) + 1 :]



#below loads text to speech models and vocoders 
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)



model.to(device)
vocoder.to(device)

#converts text to speech
def tts(text):
    # Process the text
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(input_ids,speaker_embeddings.to(device),  vocoder=vocoder)
    
    # Move the tensor to the CPU and ensure it has the correct shape
    speech = speech.squeeze().cpu()
    if len(speech.shape) == 1:
        speech = speech.unsqueeze(0)
    # Save the output to a temporary file
    output_path = "output.wav"
    torchaudio.save(output_path, speech, sample_rate=16000)
    
    return output_path

#main function that calls other 3 functions
def STT(audio):
  text = transcribe(audio)
  response = query(text)
  audio =  tts(response)
  return audio
    
#gradio interface works as frontend 
stt_gradio = gr.Interface(
    fn=STT,
    inputs=gr.Audio(sources="microphone", type="filepath", label="Speak your question"),
    outputs=gr.Audio(type="filepath", label="Generated response"),
    live=True,
    title="Audio Question to Audio Answer(Jugadu GPT4-o)",
    description="Speak a question into the microphone, and the system will generate an audio response.",
    article="""
    This application uses advanced speech processing models to convert spoken questions into spoken answers.
    Simply click on the microphone button, ask your question, and wait for the response.
    """,
    theme="huggingface"
)

# Launch the interface
stt_gradio.queue()
stt_gradio.launch(share=True, debug=True)