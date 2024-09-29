from transformers import  DPTForDepthEstimation, DPTImageProcessor
import torch


# Uses Huggingface transformers to obtain 
# Depth prediction from a image
def get_depth_prediction(pil_img, model):

    image_processor = DPTImageProcessor.from_pretrained(model)
    depth_model = DPTForDepthEstimation.from_pretrained(model)

    inputs = image_processor(images=pil_img, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    return prediction, predicted_depth
