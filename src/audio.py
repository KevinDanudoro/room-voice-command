import os
import random
import tensorflow as tf
from pydub.silence import split_on_silence

class Audio:
    def __init__(self):
        self.sample = 10000
    
    def split_audio(
        self, 
        audio,
        silence_thresh = -35, 
        min_silence_len = 200
    ):
        chunks = split_on_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        return chunks

    def pad_signal(self, signal):
        rand_pad = random.randint(0,1)
        signal = signal[:self.sample]
        zero_padding = tf.zeros([self.sample] - tf.shape(signal), dtype=tf.float32)
        signal = tf.concat([zero_padding, signal], 0) if(rand_pad == 0) else tf.concat([signal,zero_padding], 0)
        return signal

    def save_chunks(self, audio, export_dir, file_name):
        self._create_dir(export_dir)
        if(audio.duration_seconds < 0.25): return
        export_path = os.path.join(export_dir, f"{file_name}.wav")
        audio.export(export_path, format="wav")

    def _create_dir(self, path:str):
        if(os.path.exists(path=path)):
            return
        os.makedirs(path)