import yt_dlp
from pydub import AudioSegment
import os

def download_and_process_audio(url, output_path="temp_audio.wav"):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': 'downloaded_audio.%(ext)s',
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web']
                }
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        audio = AudioSegment.from_file("downloaded_audio.wav")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format="wav")
        
        if os.path.exists("downloaded_audio.wav"):
            os.remove("downloaded_audio.wav")
            
        return output_path
    except Exception as e:
        print(f"Audio processing error: {e}")
        return None
