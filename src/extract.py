import tensorflow as tf

N_SAMPLES = 6000
SAMPLE_RATE = 16000
IMAGE_SIZE = (224,224)

class Extract:

    def audio_to_mfcc(self, wav):
        wav = self._trim_audio(wav)
        stft = tf.signal.stft(wav, frame_length=256, frame_step=128)
        log_mel_spectrogram = self._convert_to_log_mel_spectrogram(stft)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfcc = self._preprocess_mfcc(mfcc)
        return mfcc


    def _trim_audio(self, wav):
        wav = wav[:N_SAMPLES]
        zero_padding = tf.zeros([N_SAMPLES] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], 0)
        return wav


    def _convert_to_log_mel_spectrogram(self, stft):
        magnitude_spectrogram = tf.abs(stft)
        num_spectrogram_bins = tf.shape(magnitude_spectrogram)[-1]
        num_mel_bins = 20
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            SAMPLE_RATE,
            100,
            7000
        )
        mel_spectrogram = tf.tensordot(
            magnitude_spectrogram,
            linear_to_mel_weight_matrix,
            1
        )
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram


    def _preprocess_mfcc(self, mfcc):
        min_mfcc = tf.reduce_min(mfcc)
        max_mfcc = tf.reduce_max(mfcc)

        normalized_mfcc = (mfcc - min_mfcc) / (max_mfcc - min_mfcc)
        rgb_mfcc = tf.tile(normalized_mfcc[:, :, tf.newaxis], [1, 1, 3])
        resized_mfcc = tf.image.resize(
            rgb_mfcc,
            IMAGE_SIZE,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return resized_mfcc
    

    def get_audio_as_tensor(self, file_name):
        file_content = tf.io.read_file(file_name)
        wav, _ = tf.audio.decode_wav(file_content, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        return wav