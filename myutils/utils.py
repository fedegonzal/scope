import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt


def load_image_as_cv2(image_path):
    """ Function printing python version. """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_as_pil(image_path):
    """ Function printing python version. """
    img = Image.open(image_path)
    return img


def load_image_as_tensor(pil_img, size=600):
    """ Function printing python version. """
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    return transform(pil_img)


def img_tensor_padded(img_tensor, patch_size):
    """ Function printing python version. """
    new_image_container = (
        img_tensor.shape[0],
        int(np.ceil(img_tensor.shape[1] / patch_size) * patch_size),
        int(np.ceil(img_tensor.shape[2] / patch_size) * patch_size),
    )
    img_paded = torch.zeros(new_image_container)
    img_paded[:, : img_tensor.shape[1], : img_tensor.shape[2]] = img_tensor

    return img_paded



def list_filenames_without_extension(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filename_without_extension = os.path.splitext(filename)[0]
            filenames.append(filename_without_extension)
    return filenames



def plot_image_grid(title, pil_img, depth_image_resized, sum_atts_resized, final_attention_map, final_att_thresholded, final_image, noise, entropy):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # set a title for the figure
    fig.suptitle(title, fontsize=16)

    axs[0, 0].imshow(pil_img)
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(depth_image_resized)
    axs[0, 1].set_title('Depth image')

    axs[0, 2].imshow(sum_atts_resized)
    axs[0, 2].set_title(f'Dino Attention (Noise: {noise:.2f} // Entropy: {entropy:.5f})')

    axs[1, 0].imshow(final_attention_map)
    axs[1, 0].set_title('Final Attention Map')

    axs[1, 1].imshow(final_att_thresholded)
    axs[1, 1].set_title('Final Attention Map Thresholded')

    axs[1, 2].imshow(final_image)
    axs[1, 2].set_title('Final Image')

    #plt.show()
