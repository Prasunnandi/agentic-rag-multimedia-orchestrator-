import whisper

class Transcriber:
    def __init__(self, model_size="tiny"):
        # Using 'tiny' model so it works well on Streamlit Free without heavy memory usage
        self.model = whisper.load_model(model_size)

    def transcribe_whisper(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]
