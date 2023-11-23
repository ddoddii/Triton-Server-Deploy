import triton_python_backend_utils as pb_utils
import torch 
import numpy

class TritonPythonModel:

    def initialize(self, args):
        # Load your model here
        model_path = "/models/dreamlike-photoreal/1/model.ckpt"
        self.model = torch.load(model_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            input_text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text = input_text_tensor.as_numpy()

            # Perform inference
            output_image = self.model.generate_image(input_text)  # Placeholder method

            # Convert the output to the appropriate format for Triton response
            output_image_tensor = pb_utils.Tensor("OUTPUT_IMAGE", output_image.astype(numpy.uint8))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_image_tensor])
            responses.append(inference_response)

        return responses
