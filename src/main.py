import tkinter as tk
import numpy as np

from preprocess import librosa_split
from microphone import Microphone
from extract import Extract
from model import Model
from thread import CustomThread


def capture_audio_from_mic():
    webcam = Microphone(device_id=4)
    audio = webcam.record(duration=5)
    return audio

def split_audio(audio):
    audio_slices = librosa_split(audio)
    return audio_slices

def extract_audio_to_mfcc(audio):
    extractor = Extract()
    mfcc = extractor.librosa_audio_to_mfcc(np.expand_dims(audio, axis=0))
    return mfcc

def predict_audio(model, mfccs):
    result = model.predict_onnx(mfccs)
    return result

def show_predict_result(result):
    window = tk.Tk()
    window.geometry("500x300") 
    label = tk.Label(window, text=f"Kode Ruangan: {result}")
    custom_font = ("Arial", 32) 
    label.config(font=custom_font)
    label.pack(expand=True, fill="both")
    window.mainloop()

def main():
    full_audio = capture_audio_from_mic()
    audio_slices = split_audio(full_audio)

    threads = []
    for audio in audio_slices:
        t_extract = CustomThread(target=extract_audio_to_mfcc, args=(audio,))
        t_extract.start()
        threads.append(t_extract)

    resnet = Model("model/resnet_aug7.onnx")
    print("selesai load model")

    mfccs = [thread.join() for thread in threads]
    print(mfccs)

    result = predict_audio(resnet, mfccs)
    show_predict_result(result)

if __name__ == "__main__":
    main()