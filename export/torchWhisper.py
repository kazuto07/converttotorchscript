import torch
from transformers import Wav2Vec2ForCTC, WhisperProcessor
import torchaudio
tokenizer=WhisperProcessor.from_pretrained("openai/whisper-large-v2")
loaded_model = torch.jit.load("speech_to_text_model.pt")
loaded_model.eval()
dummy_input = torch.randn(1, 100000) 
print(dummy_input)
waveform, sample_rate = torchaudio.load("export/output.wav")
print(waveform)
with torch.no_grad():
    output = loaded_model(waveform)
print(sample_rate)
print(output)
output_tensor = output['logits']

# Get the predicted token IDs
predicted_token_ids = torch.argmax(output_tensor, dim=-1)

# Decode the predicted token IDs
decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

print(decoded_text)