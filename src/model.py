import tensorflow as tf
import onnxruntime as ort
import numpy as np

class Model:
    def __init__(self, path,  classnames=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'f', 'noise']):
        self.model = self._load_onnx(path)
        self.classnames = classnames
    
    def _load_onnx(self,path):
        providers = ['CPUExecutionProvider','CUDAExecutionProvider']
        ort_sess = ort.InferenceSession(path, providers=providers)
        return ort_sess

    def predict_onnx(self, data):
        predicted_batch = self.model.run(None, {"input": data})
        predicted_batch = tf.squeeze(predicted_batch).numpy()
        predicted_classes = [self.classnames[np.argmax(pred)] if(np.max(pred) > 0.9) else None for pred in predicted_batch]
        return predicted_classes