import os
import gc
os.environ['PA_ALSA_DEBUG'] = '0'

from audio import Audio
from microphone import Microphone


# Mendapatkan audio dari microphone
webcam = Microphone(device_index=4)
audio = webcam.capture_audio(access_time=30)

del webcam
gc.collect()


room_voice = Audio(audio)
# room_voice.save_chunks(audio, export_dir="audio/testset", file_name="ftest")

audio_slices = room_voice.split_audio()
for i, chunk in enumerate(audio_slices): 
    room_voice.save_chunks(
        audio=chunk, 
        export_dir="audio/dataset/f",
        file_name=f"test-{i}" 
    )

del room_voice
gc.collect()

