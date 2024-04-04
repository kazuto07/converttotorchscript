import torch
from transformers import WhisperForCausalLM, WhisperProcessor,WhisperConfig
import torchaudio

import librosa
# Load the pre-trained speech-to-text model and processor
model = WhisperForCausalLM.from_pretrained("openai/whisper-tiny", torchscript=True)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
waveform,sample_rate=librosa.load("export/extracted_audio..mp3")
# Preprocess the audio waveform
input_features = processor(waveform,return_tensors="pt").input_features
print(input_features)
# Ensure the input features have the correct shape
#dummmy_input=torch.randn(1,80,3000)
model.config.forced_decoder_ids = None
decoder_input_id = torch.zeros_like(input_features)
#de =torch.long(input_features)
# Verify input dimensions
# Trace the model with the preprocessed input
model.eval()
traced_model = torch.jit.trace(model, decoder_input_id)

# Save the traced model to a file
traced_model.save("speech_to_text_model.pt")
