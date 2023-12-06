import tkinter as tk
import numpy as np

from pyaudio import PyAudio, paFloat32
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from preprocess import librosa_split
from extract import Extract
from model import Model
from thread import CustomThread
import soundfile as sf

SAMPLE_RATE = 16000

def init_mic():
    p = PyAudio()
    stream = p.open(format=paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
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
        if(audio.shape[0] % SAMPLE_RATE <= 50):
            label.configure(text=f"Merekam Suara Selama {5 - (audio.shape[0] // SAMPLE_RATE)} detik")
        audio_stream = np.frombuffer(stream.read(3200), dtype=np.float32)
        audio = np.concatenate((audio, audio_stream))
        if(audio.shape[0] < 5 * SAMPLE_RATE): continue
        stream.stop_stream()
        label.configure(text="Memproses Suara...")

        result = callback(audio[:5 * SAMPLE_RATE])
        label.configure(text=f"Kode Ruangan: {result[0]}{result[1]}.{result[2]}")
        loop = False

def extract_audio_to_mfcc(audio):
    extractor = Extract()
    mfcc = extractor.librosa_audio_to_mfcc(np.expand_dims(audio, axis=0))
    return mfcc

def predict_audio(model, mfccs):
    result = model.predict_onnx(mfccs)
    return result

def main(full_audio):
    audio_slices = librosa_split(full_audio)
    threads = []
    for i, audio in enumerate(audio_slices):
        sf.write(f"audio/testset/test_kevin_f01h_{i+1}.wav", audio.numpy(), SAMPLE_RATE)
        t_extract = CustomThread(target=extract_audio_to_mfcc, args=(audio,))
        t_extract.start()
        threads.append(t_extract)

    resnet = Model("model/resnet_aug7.onnx")
    mfccs = [thread.join() for thread in threads]
    result = predict_audio(resnet, mfccs)
    result = [res for res in result if(res != 'noise')]
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