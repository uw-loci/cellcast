from pathlib import Path
import tf2onnx
import onnx
import tensorflow as tf
from stardist.models import StarDist3D

# create the StarDist2D model with either pre-trained or custom weights
# sd_model = StarDist2D(None, name="custom", basedir=Path("path/to/model_dir"))
sd_model = StarDist3D.from_pretrained("3D_demo")
model = sd_model.keras_model

# create a fixed input specification
input_spec = tf.TensorSpec((1, 512, 512, 512, 1), tf.float32, name="input")

# convert to ONNX
output_path = "onnx_models/staridst3d_model.onnx"
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[input_spec],
    opset=18, # Burn recommends models use opset 16 or higher for best compatibility
    output_path=output_path
)
print("[INFO]: Model converted to ONNX.")

# verify the ONNX model
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("[INFO]: ONNX model is valid.") 
