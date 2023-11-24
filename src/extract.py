import tensorflow as tf
import librosa

N_SAMPLES = 10000
SAMPLE_RATE = 16000
IMAGE_SIZE = (128,128)

class Extract:

  def librosa_audio_to_mfcc(self, audio):
    audio = tf.squeeze(audio, axis=0).numpy()
    mfcc = librosa.feature.mfcc(
      y=audio, sr=SAMPLE_RATE, n_mfcc=13, norm='ortho', n_fft=1024, win_length=1024, hop_length=512, n_mels=80, fmax=8000, fmin=80
    )

    vmin = tf.reduce_min(mfcc)
    vmax = tf.reduce_max(mfcc)

    normalize = (mfcc - vmin) / (vmax - vmin)
    rgb = tf.tile(normalize[:, :, tf.newaxis], [1, 1, 3])
    resize = tf.image.resize(rgb, IMAGE_SIZE, method='nearest')
    return resize
