""" SSL utilities """
from transformers import  DPTForDepthEstimation, DPTImageProcessor
import torch
import numpy as np
import cv2

from scipy.optimize import linear_sum_assignment

from torch import nn

from dinov1.vision_transformer import vit_small

from myutils.pascal_voc import convert_ground_truth_voc2007_to_list, get_ground_truth_voc2007, draw_ground_truth_voc2007
from myutils.utils import img_tensor_padded, load_image_as_cv2, load_image_as_pil, load_image_as_tensor



def create_mask_fix_depth(depth_normalized, mode='vertical'):
    """ Function printing python version. """

    # Define the dimensions of the mask
    h, w = depth_normalized.shape

    # Define how strong the color grading should be
    strength = depth_normalized.max() / 1.5

    # Create a meshgrid for x and y coordinates
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # Create the mask with vertical color grading from 0 to strength

    if mode == 'vertical':
        mask = ((y - 0.5) * (strength*2)).astype(np.uint8)
        mask[y < 0.5] = 0
    
    elif mode == 'horizontal':
        mask = ((x - 0.5) * (strength*2)).astype(np.uint8)
        mask[x < 0.5] = 0

    return mask


def preprocess_depth(depth_normalized, prediction, predicted_depth):

    #plt.imshow(depth_normalized)
    #plt.title("Depth prediction")
    #plt.show()

    # print min, max, and mean depth values
    #print("Min depth:", depth_normalized.min())
    #print("Max depth:", depth_normalized.max())
    #print("Mean depth:", depth_normalized.mean())
    #print("Std:", depth_normalized.std())

    # apply the mask to the depth
    mask = create_mask_fix_depth(depth_normalized, mode='vertical')
    depth_normalized = depth_normalized.astype(np.int32) - mask.astype(np.int32)
    depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)

    #plt.imshow(depth_normalized)
    #plt.title("Depth prediction with mask")
    #plt.show()

    out_softmin = torch.nn.functional.softmin(prediction.squeeze(), dim=-1)
    out_softmax = torch.nn.functional.softmax(predicted_depth.squeeze(), dim=-1)
    out_group_norm = torch.nn.functional.group_norm(predicted_depth.squeeze(), 1)
    out_normalize = torch.nn.functional.normalize(predicted_depth.squeeze(), p=2, dim=-1)

    depth_layers = 21

    out_layers = cv2.normalize(depth_normalized, None, 0, depth_layers, cv2.NORM_MINMAX)
    out_layers = out_layers.astype(np.uint8)

    #plt.show()

    # threshold out_layers image, values < 5 will be set to 0
    out_layers[out_layers < depth_layers/3] = 0
    depth = out_layers * prediction.squeeze().cpu().numpy()

    return depth


def load_ssl_model(patch_size, checkpoint, device, img_size=None):
    """ Function printing python version. """

    checkpoint = torch.load(checkpoint, map_location=device)

    # Load the model
    model = vit_small(
        patch_size=patch_size,
        init_values=1.0,
        block_chunks=0
    )


    for p in model.parameters():
        p.requires_grad = False

    state_dict = model.load_state_dict(checkpoint)

    return model



def get_attentions(model, img_paded, patch_size):

    # Size for transformers
    w_featmap = img_paded.shape[-2] // patch_size
    h_featmap = img_paded.shape[-1] // patch_size


    with torch.no_grad():
        attentions = model.get_last_selfattention(img_paded[None, :, :, :])

    # we keep only the output patch attention
    # for every patch
    nh = attentions.shape[1]  # Number of heads
    atts = attentions[0, :, 0, 1:].reshape(nh, -1)

    atts = atts.reshape(nh, w_featmap, h_featmap)

    # resize to image size
    atts = nn.functional.interpolate(atts.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    return atts




def calculate_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate intersection area
    if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
        inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        inter_area = 0

    # Calculate union area
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    return iou

def calculate_iou_matrix(pred_boxes, gt_boxes):
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    return iou_matrix



def get_boxes(final):

    # Find the contours
    contours, hierarchy = cv2.findContours(final.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    colors = [
        (0, 0, 255), # blue
        (0, 255, 0), # green
        (255, 0, 0), # red
        (255, 255, 0), # cyan
        (255, 0, 255), # magenta
        (0, 255, 255), # yellow
    ]

    the_contours = []

    # Draw the contours with different colors
    for i, contour in enumerate(contours):
        
        # Using hierarchy to filter out the inner contours
        if hierarchy[0][i][3] != -1:
            continue

        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < 1000:
            continue

        # Approximate the contour with a rectangle
        x, y, w, h = cv2.boundingRect(contour)    
        the_contours.append([x, y, w, h])

    return the_contours

'''
def draw_boxes(contours, img):
    colors = [
        (0, 0, 255), # blue
        (0, 255, 0), # green
        (255, 0, 0), # red
        (255, 255, 0), # cyan
        (255, 0, 255), # magenta
        (0, 255, 255), # yellow
    ]

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        thickness = 3

        x, y, w, h = contour
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    return img
'''


def process_prediction(prediction, predicted_depth, patch_size, img_paded, device):

    # preprocess the depth prediction
    depth = prediction.squeeze().cpu().numpy()
    depth_normalized = (depth * 255 / np.max(depth)).astype("uint8")
    
    # ToDo: review this function
    #depth = preprocess_depth(depth_normalized, prediction, predicted_depth)
    depth = depth_normalized

    model = load_ssl_model(patch_size, 'pretrained/dino_deitsmall16_pretrain.pth', device, None)
    model.to(device)
    model.eval()

    atts = get_attentions(model, img_paded, patch_size)

    # Mean attentions into one map
    #sum_atts = atts.mean(0)
    sum_atts = atts.sum(0)
    sum_atts_resized = cv2.resize(sum_atts, (depth.shape[1],depth.shape[0]))

    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    sum_atts_resized_norm = cv2.normalize(sum_atts_resized, None, 0, 1, cv2.NORM_MINMAX)

    depth_weight = 0.2
    att_weight = 0.8

    prefinal =  (depth_norm * sum_atts_resized_norm) + (depth_norm * depth_weight) + (sum_atts_resized_norm * att_weight)

    #prefinal = cv2.normalize(prefinal, None, 0, 1, cv2.NORM_MINMAX)

    #print(
    #    "Prefinal ->",
    #    "min:", prefinal.min(), 
    #    "mean:", prefinal.mean(), 
    #    "max:", prefinal.max(),
    #    "std:", prefinal.std(),
    #)

    # remove values lower than std
    final = prefinal.copy()

    final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)

    #print(
    #    "Final ->",
    #    "min:", final.min(), 
    #    "mean:", final.mean(), 
    #    "max:", final.max(),
    #    "std:", final.std(),
    #)


    final = prefinal.copy()

    final[final > prefinal.std()] = 255
    final[final <= prefinal.std()] = 0

    final = final.astype(np.uint8)

    return final





def predict_main_objects(image_path, annotation_path, device, return_image=False):
    # Load an image to find its main objects

    img = load_image_as_cv2(image_path)
    pil_img = load_image_as_pil(image_path)

    # Load and transform the selected image
    img_tensor = load_image_as_tensor(pil_img).to(device)

    # Padding the image with zeros to fit multiple of patch-size
    patch_size = 16
    img_paded = img_tensor_padded(img_tensor, patch_size).to(device)

    ground_truth = get_ground_truth_voc2007(annotation_path)
    ground_truth_img = draw_ground_truth_voc2007(img, ground_truth)

    # get the depth prediction
    prediction, predicted_depth = get_depth_prediction(pil_img)
    # preprocess the depth prediction
    processed_prediction = process_prediction(
        prediction, predicted_depth, patch_size, img_paded, device
    )

    predicted_boxes = get_boxes(processed_prediction)

    # Display the output image with the contours and the ground truth
    if return_image:
        final_image = get_output_image(img, predicted_boxes, ground_truth)
    else:
        final_image = None

    return predicted_boxes, convert_ground_truth_voc2007_to_list(ground_truth), final_image
