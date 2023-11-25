from pyaudio import PyAudio, paFloat32
from numpy import array, concatenate, frombuffer, float32

class Microphone:
    def __init__(self, rate=16000, frames_per_buffer=3200, device_id=11):
        self.rate=rate
        self.chunk = frames_per_buffer
        self.device_id=device_id

    def record(self, duration=5):
        # Inisialisasi PyAudio
        p = PyAudio()

        # Membuka stream untuk perekaman audio
        stream = p.open(format=paFloat32,
                        channels=1,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device_id)

        # Membaca data audio dari microphone
        print("Mulai merekam")
        audio = array([])
        for _ in range(0, int(self.rate/self.chunk * duration)):
            audio_stream = frombuffer(stream.read(self.chunk), dtype=float32)
            audio = concatenate((audio, audio_stream))
        print("Selesai merekam")

        # Menutup stream dan PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        return audio