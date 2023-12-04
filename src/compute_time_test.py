import tkinter as tk
from time import time

import numpy as np
from pyaudio import PyAudio, paFloat32

from preprocess import librosa_split
from extract import Extract
from model import Model
from thread import CustomThread

def init_mic():
    p = PyAudio()
    stream = p.open(format=paFloat32,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=3200,
                    input_device_index=4)
    stream.stop_stream()
    return p, stream

def listen_mic(stream, callback, label):
    loop = True
    stream.start_stream()
    audio = np.array([])
    while loop:
        if(audio.shape[0] % 16000 <= 50):
            label.configure(text=f"Merekam Suara Selama {5 - (audio.shape[0] // 16000)} detik")
        audio_stream = np.frombuffer(stream.read(3200), dtype=np.float32)
        audio = np.concatenate((audio, audio_stream))
        if(audio.shape[0] < 80000): continue
        stream.stop_stream()
        label.configure(text="Memproses Suara...")

        result = callback(audio[:80000])
        label.configure(text=f"Kode Ruangan: {result}")
        loop = False

def extract_audio_to_mfcc(audio):
    extractor = Extract()
    mfcc = extractor.librosa_audio_to_mfcc(np.expand_dims(audio, axis=0))
    return mfcc

def predict_audio(model, mfccs):
    predict_start = time()
    result = model.predict_onnx(mfccs)
    predict_end = time()

    print(f"-----// Waktu Prediksi ResNet Model: {predict_end - predict_start} //-----")
    return result

def main(full_audio):
    audio_slices = librosa_split(full_audio)

    mfcc_start = time()

    threads = []
    for audio in audio_slices:
        t_extract = CustomThread(target=extract_audio_to_mfcc, args=(audio,))
        t_extract.start()
        threads.append(t_extract)

    resnet = Model("model/resnet_aug7.onnx")
    mfccs = [thread.join() for thread in threads]
    
    mfcc_end = time()
    print(f"-----// Waktu ekstraksi MFCC: {mfcc_end - mfcc_start} //-----")

    result = predict_audio(resnet, mfccs)
    return result

def terminate_process(p, stream, window):
    stream.stop_stream()
    stream.close()
    p.terminate()
    window.destroy()


if __name__ == "__main__":
    p, stream = init_mic()

    window = tk.Tk()
    window.attributes('-fullscreen', True)
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)

    label = tk.Label(window, text=f"Sistem Pengenalan Suara Kode Ruangan")
    custom_font = ("Arial", 28) 
    label.config(font=custom_font)
    label.grid(column=0, row=0, columnspan=2)

    button_frame = tk.Frame(window)
    button_frame.grid(column=0, row=1, columnspan=2)
    button_continue = tk.Button(
        button_frame, 
        text="Mulai", 
        command=lambda: CustomThread(target=listen_mic, args=(stream, main, label)).start()
    )
    button_continue.grid(column=0, row=1)

    button_terminate = tk.Button(
        button_frame, 
        text="Keluar", 
        command=lambda: terminate_process(p, stream, window)
    )
    button_terminate.grid(column=1, row=1)

    window.mainloop()