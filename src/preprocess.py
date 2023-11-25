import sklearn
import tensorflow as tf
import numpy as np

def librosa_split(audio, sr=16000):
  import librosa
  import noisereduce as nr

  reduced_audio = nr.reduce_noise(y=audio, sr=sr, thresh_n_mult_nonstationary=2,stationary=False)

  stft = librosa.stft(reduced_audio,n_fft=1024, hop_length=256, win_length=1024)
  db_levels=librosa.amplitude_to_db(np.abs(stft), ref=np.max)
  std_db = np.std(np.abs(db_levels))
  mean_db = np.mean(np.abs(db_levels))

  AUDIO_SAMPLES = 5 * sr    
  CODE_AUDIO_SAMPLES = AUDIO_SAMPLES * (4/8)
  top_db = ((mean_db-std_db*(4/8))*(CODE_AUDIO_SAMPLES/AUDIO_SAMPLES))

  silence = librosa.effects.split(reduced_audio, top_db=top_db, frame_length=2048, hop_length=512)

  audio_slices = [audio[chunk[0]:chunk[1]] for chunk in silence]
  fix_sound = [chunk for chunk in audio_slices if(chunk.shape[0] >= 6000)]
  pad_audio = [trim_audio(chunk) for chunk in fix_sound]
  return pad_audio


def trim_audio(wav, sample=10000):
  import random
  
  rand_trim = random.randint(0,1)
  wav = wav[:sample]
  zero_padding = tf.zeros([sample] - tf.shape(wav), dtype=tf.float32)
  wav = tf.concat([zero_padding, wav], 0) if(rand_trim == 0) else tf.concat([wav,zero_padding], 0)
  return wav