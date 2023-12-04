from audio import Audio
from microphone import Microphone

webcam = Microphone(device_index=4)
audio = webcam.record(access_time=30)

room_voice = Audio()
audio_slices = room_voice.split_audio(audio)
for i, chunk in enumerate(audio_slices): 
    pad_chunk = room_voice.pad_signal(chunk)
    room_voice.save_chunks(
        audio=pad_chunk, 
        export_dir="audio/dataset/f",
        file_name=f"test-{i}" 
    )