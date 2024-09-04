from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import os
import uuid

app = Flask(__name__)

# Initialize TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.route('/generate-voice', methods=['POST'])
def generate_voice():

    text = request.form.get('text')
    language = request.form.get('language', 'en')
    speaker_wav = request.files.get('speaker_wav')

    if not text or not speaker_wav:
        return jsonify({"error": "Missing required parameters"}), 400

        # Save the uploaded speaker wav file temporarily
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    speaker_wav_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{speaker_wav.filename}")
    speaker_wav.save(speaker_wav_path)

        # Generate voice
    output_file = os.path.join(temp_dir, f"{uuid.uuid4()}_output.wav")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path=output_file)

        # Return the generated file
    return send_file(output_file, as_attachment=True, mimetype='audio/wav')
    

if __name__ == '__main__':
    app.run()
