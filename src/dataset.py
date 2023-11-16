import os
import gc
os.environ['PA_ALSA_DEBUG'] = '0'

from audio import Audio
from microphone import Microphone


# Mendapatkan audio dari microphone
webcam = Microphone(device_index=4)
audio = webcam.capture_audio(access_time=5)

del webcam
gc.collect()


# Mereduksi noise dan membagi audio berdasarkan fase hening
room_voice = Audio(audio)
room_voice.save_chunks(
    audio=audio,
    export_dir="audio/testset", 
    file_name=f"f64"
)

del room_voice
gc.collect()

