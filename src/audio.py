import os
# from pydub.silence import split_on_silence
# from pydub.audio_segment import AudioSegment
# from noisereduce import reduce_noise
# import numpy as np

class Audio:
    def __init__(self, audio):
        self.audio = audio
    

    # def split_audio(
    #     self, 
    #     silence_thresh = -35, 
    #     min_silence_len = 200
    # ):
    #     chunks = split_on_silence(
    #         self.audio, 
    #         min_silence_len=min_silence_len, 
    #         silence_thresh=silence_thresh
    #     )
    #     return chunks


    def save_chunks(self, audio, export_dir, file_name):
        self._create_dir(export_dir)
        if(audio.duration_seconds < 0.25): return
        export_path = os.path.join(export_dir, f"{file_name}.wav")
        audio.export(export_path, format="wav")


    # def noise_reduce(self):
    #     samples = np.array(self.audio.get_array_of_samples())
    #     sr = self.audio.frame_rate
    #     reduced_noise = reduce_noise(samples, sr=sr)
    #     reduced_audio = AudioSegment(
    #         reduced_noise.tobytes(),
    #         frame_rate=sr,
    #         sample_width=self.audio.sample_width,
    #         channels=self.audio.channels
    #     )
    #     return reduced_audio
    

    def _create_dir(self, path:str):
        if(os.path.exists(path=path)):
            return
        os.makedirs(path)