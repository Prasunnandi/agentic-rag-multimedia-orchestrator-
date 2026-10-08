import whisper
import os

class VideoTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, file_path):
        print(f"Transcribing {file_path}...")
        if not os.path.exists(file_path):
            return "File not found. Please provide a valid media file."
        result = self.model.transcribe(file_path)
        return result["text"]

if __name__ == "__main__":
    pass
