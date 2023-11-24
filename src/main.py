import tkinter as tk

from preprocess import *
from model import Model
from extract import Extract
from microphone import Microphone


# Mendapatkan audio dari microphone
webcam = Microphone(device_index=11)
audio = webcam.capture_audio(access_time=5)
del webcam

extractor = Extract()
audio_slices = librosa_split(audio)
mfccs = [extractor.librosa_audio_to_mfcc(np.expand_dims(audio, axis=0)) for audio in audio_slices]

resnet = Model("model/resnet_aug7.onnx")
predicted_results = resnet.predict_onnx(mfccs)
classname = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'f']
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
