import whisper

def transcribe_audio(audio_file_path, model_size="tiny"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file_path)
    return result['text']
