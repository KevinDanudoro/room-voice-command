import tensorflow as tf
import numpy as np
import onnxruntime as ort

class Model:
    def __init__(self, model_path):
        self.model_path = model_path

    def _load_model(self):
        loaded_model = tf.keras.models.load_model(self.model_path)
        return loaded_model
    
    def predict(self, data):
        loaded_model = self._load_model()
        predicted = loaded_model.predict(data)
        predicted_ids = np.argmax(predicted, axis=-1)
        return predicted_ids
    
    def _load_onnx(self):
        providers = ['CPUExecutionProvider','CUDAExecutionProvider']
        ort_sess = ort.InferenceSession(self.model_path, providers=providers)
        return ort_sess

    def predict_onnx(self, data):
        ort_sess = self._load_onnx()
        predicted_batch = ort_sess.run(None, {"input": data})
        predicted_batch = tf.squeeze(predicted_batch).numpy()
        predicted_ids = np.argmax(predicted_batch, axis=-1)
        return predicted_ids