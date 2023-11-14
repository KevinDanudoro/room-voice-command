import wave
import os
from pydub import AudioSegment
import pyaudio
from time import time

class Microphone: 
    def __init__(self, device_index):
        self.device_index = device_index


    def capture_audio(
            self, 
            access_time = 5, 
            frame_per_buffer = 3200, 
            format = pyaudio.paInt16, 
            channels = 1, 
            rate = 16000,
            output_path = "output.wav"
        ):
        frames, sample_size = self._record(access_time, frame_per_buffer, format, channels, rate)
        
        self._save_audio(path=output_path, frames=frames, sample_size=sample_size, channels=channels, rate=rate)
        audio = AudioSegment.from_wav(file=output_path)
        self._delete_audio(path=output_path)

        return audio

    def _record(
            self, 
            access_time, 
            frame_per_buffer, 
            format, 
            channels, 
            rate,
        ):
        p = pyaudio.PyAudio()
        sample_size = p.get_sample_size(format)
        stream = p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=frame_per_buffer,
            input_device_index=self.device_index
        )

        print("Start Recording")
        frames = []
        start = time()
        for _ in range(0, int(rate/frame_per_buffer*access_time)):
            chunk = stream.read(frame_per_buffer)
            frames.append(chunk)
            if(time() - start > access_time - 2): print('Stop Talking')

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording Stopped")

        return frames, sample_size


    def _save_audio(self, path, frames, sample_size, channels, rate):
        obj = wave.open(path, "wb")
        obj.setnchannels(channels)
        obj.setsampwidth(sample_size)
        obj.setframerate(rate)
        obj.writeframes(b"".join(frames))
        obj.close()


    def _delete_audio(self, path):
        if(os.path.exists(path)):
            os.remove(path)
        else:
            print(f"File {path} does'nt exist")