import tensorflow as tf

N_SAMPLES = 10000
SAMPLE_RATE = 16000
IMAGE_SIZE = (128,128)

class Extract:

    def audio_to_mfcc(self, wav):
        wav = self._trim_audio(wav)
        stft = tf.signal.stft(wav, frame_length=1024, frame_step=256, fft_length=1024)
        log_mel_spectrogram = self._convert_to_log_mel_spectrogram(stft)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :13]
        mfcc = self._preprocess_mfcc(mfcc)
        return mfcc
    

    def get_audio_as_tensor(self, file_name):
        file_content = tf.io.read_file(file_name)
        wav, _ = tf.audio.decode_wav(file_content, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        return wav


    def _trim_audio(self, wav):
        wav = wav[:N_SAMPLES]
        zero_padding = tf.zeros([N_SAMPLES] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], 0)
        return wav


    def _convert_to_log_mel_spectrogram(self, stft):
        spectrograms = tf.abs(stft)
        num_spectrogram_bins = stft.shape.as_list()[-1]
        num_mel_bins = 80

        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins,
          num_spectrogram_bins,
          SAMPLE_RATE,
          lower_edge_hertz,
          upper_edge_hertz
        )

        mel_spectrogram = tf.tensordot(
          spectrograms, linear_to_mel_weight_matrix, 1
        )
        mel_spectrogram.set_shape(
          spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]
          )
        )
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram


    def _preprocess_mfcc(self, mfcc):
        vmin = tf.reduce_min(mfcc)
        vmax = tf.reduce_max(mfcc)

        normalize = (mfcc - vmin) / (vmax - vmin)
        rgb = tf.tile(normalize[:, :, tf.newaxis], [1, 1, 3])
        # rgb = tf.expand_dims(normalize, -1)
        resize = tf.image.resize(rgb, IMAGE_SIZE, method='nearest')
        flip_up_down = tf.image.flip_up_down(resize)

        return flip_up_down