from audio import Audio
from model import Model
from extract import Extract
from microphone import Microphone

import tensorflow as tf
import tkinter as tk
import os
import gc

TEST_DIR = "audio/testset"


# # Mendapatkan audio dari microphone
# webcam = Microphone(device_index=11)
# audio = webcam.capture_audio(access_time=12)

# del webcam
# gc.collect()


# # Mereduksi noise dan membagi audio berdasarkan fase hening
# room_voice = Audio(audio)
# audio = room_voice.noise_reduce()
# chunks = room_voice.split_audio()
# for i, chunk in enumerate(chunks): 
#     room_voice.save_chunks(
#         audio=chunk, 
#         export_dir=TEST_DIR, 
#         file_name=f"test-{i}" 
#     )

# del room_voice
# gc.collect()

extractor = Extract()
file_names = [os.path.join(TEST_DIR, file) for file in sorted(os.listdir(TEST_DIR))]
data = tf.data.Dataset.from_tensor_slices(file_names)
data = data.map(extractor.get_audio_as_tensor)
data = data.map(extractor.audio_to_mfcc)
data = data.batch(8)
del extractor

resnet = Model("model/resnet2.keras")
predicted_results = resnet.predict(data)
classname = ['9', '6', '5', '1', '0', '3', '7', '2', '8', '4', 'F']
predicted_classes = [classname[id] for id in predicted_results]
del resnet

window = tk.Tk()
window.geometry("500x300") 
label = tk.Label(window, text=f"Kode Ruangan: {predicted_classes}")
custom_font = ("Arial", 32) 
label.config(font=custom_font)
label.pack(expand=True, fill="both")
window.mainloop()
del window
