import os
from pydub.silence import split_on_silence

class Audio:
    def __init__(self, audio):
        self.audio = audio
    

    def split_audio(
        self, 
        silence_thresh = -35, 
        min_silence_len = 200
    ):
        chunks = split_on_silence(
            self.audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        return chunks


    def save_chunks(self, audio, export_dir, file_name):
        self._create_dir(export_dir)
        if(audio.duration_seconds < 0.25): return
        export_path = os.path.join(export_dir, f"{file_name}.wav")
        audio.export(export_path, format="wav")
    

    def _create_dir(self, path:str):
        if(os.path.exists(path=path)):
            return
        os.makedirs(path)