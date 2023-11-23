import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import safetensors.torch

model_name = "dreamlike-photoreal-2.0"

# Create a Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare and serialize the input data
text_input = "Your input text here"
text_tensor = np.array([text_input], dtype=object)
safe_text_tensor = safetensors.torch.serialize(text_tensor)

# Convert input data to Triton's InferInput object
input_tensor = httpclient.InferInput("INPUT_TEXT", safe_text_tensor.shape, np_to_triton_dtype(safe_text_tensor.dtype))
input_tensor.set_data_from_numpy(safe_text_tensor)

# Create the inference request
request = client.infer(model_name, model_version="1", inputs=[input_tensor])

# Get the result, deserialize it and process
result_safe_tensor = request.as_numpy("OUTPUT_IMAGE")
result = safetensors.torch.deserialize(result_safe_tensor)

# Now, 'result' contains your output image data
# Process this as required for your application
