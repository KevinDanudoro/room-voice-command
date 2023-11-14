import os
import gc
os.environ['PA_ALSA_DEBUG'] = '0'

from pydub.playback import play

from audio import Audio
from microphone import Microphone


# Mendapatkan audio dari microphone
webcam = Microphone(device_index=11)
audio = webcam.capture_audio(access_time=30)

del webcam
gc.collect()


# Mereduksi noise dan membagi audio berdasarkan fase hening
room_voice = Audio(audio)
audio = room_voice.noise_reduce()
chunks = room_voice.split_audio()
for i, chunk in enumerate(chunks): 
    room_voice.save_chunks(
        audio=chunk, 
        export_dir="audio/dataset/berhenti", 
        file_name=f"wildan-{i}" 
    )

del room_voice
gc.collect()

