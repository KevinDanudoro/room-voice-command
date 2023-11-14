import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("model/resnet2.keras")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model/" + "resnet2" + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=16, output_path=output_path)