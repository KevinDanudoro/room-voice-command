import tensorflow as tf
import numpy as np

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