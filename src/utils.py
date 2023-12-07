import random
import torch
import numpy as np
import cv2

def set_random_seed(random_seed: int = 42) -> None:

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cv2.setRNGSeed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt
import matplotlib.path as path
import numpy as np
import os
import cv2
import yaml
from peft import get_peft_model_state_dict
from safetensors.torch import save_file


# Function to parse the text label and extract polygon information
def parse_labels(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()

    polygons = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            coordinates = [float(coord) for coord in parts[1:]]
            num_points = len(coordinates) // 2  # Number of (x, y) pairs
            polygon = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
            polygons.append((class_id, polygon))

    return polygons

# Function to plot bounding boxes
def plot_polygons(image_path, polygons):
    # Load the image if needed
    image = plt.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Generate unique colors for classes
    class_ids = set([class_id for class_id, _ in polygons])
    num_classes = len(class_ids)
    colors = generate_colors(num_classes)

    # Plot each polygon
    for class_id, polygon in polygons:
        x, y = zip(*polygon)
        color = colors[class_id % num_classes]
        polygon = plt.Polygon(np.c_[x, y], linewidth=1, edgecolor=color, facecolor='none', label=f'Class {class_id}')
        ax.add_patch(polygon)

    # Display the image
    ax.imshow(image)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.legend()

    # Show the plot
    plt.show()

# Function to generate unique colors based on class ID
def generate_colors(num_classes):
    colormap = plt.cm.get_cmap('tab20', num_classes)
    colors = [colormap(i) for i in range(num_classes)]
    return colors

# Function to scale polygons to match an image size
def scale_polygons(polygons, image_size):
    scaled_polygons = []

    for class_id, polygon in polygons:
        # Scale the polygon coordinates
        scaled_polygon = [(x * image_size[0], y * image_size[1]) for x, y in polygon]
        scaled_polygons.append((class_id, scaled_polygon))

    return scaled_polygons

def modify_mask(mask, scaling_pixels=1):
    modified_mask = mask.copy()

    if scaling_pixels > 0:
        for s in range(1, scaling_pixels+1):
            # Create masks for shifting in all four directions
            up_mask = np.roll(mask, s, axis=0)
            down_mask = np.roll(mask, -s, axis=0)
            left_mask = np.roll(mask, s, axis=1)
            right_mask = np.roll(mask, -s, axis=1)

            # Use logical OR to combine the shifted masks
            modified_mask = np.logical_or(modified_mask, up_mask)
            modified_mask = np.logical_or(modified_mask, down_mask)
            modified_mask = np.logical_or(modified_mask, left_mask)
            modified_mask = np.logical_or(modified_mask, right_mask)

    elif scaling_pixels < 0:
        for s in range(1, abs(scaling_pixels)+1):
            # Create masks for shifting in all four directions
            up_mask = np.roll(mask, -s, axis=0)
            down_mask = np.roll(mask, s, axis=0)
            left_mask = np.roll(mask, -s, axis=1)
            right_mask = np.roll(mask, s, axis=1)

            # Use logical AND to combine the shifted masks
            modified_mask = np.logical_and(modified_mask, up_mask)
            modified_mask = np.logical_and(modified_mask, down_mask)
            modified_mask = np.logical_and(modified_mask, left_mask)
            modified_mask = np.logical_and(modified_mask, right_mask)

    return modified_mask.astype(np.uint8)

# generate mask from polygons
def generate_masks(polygons, image_size, num_samples=None, scaling_pixels=None, labels_filter=None):

    w, h = image_size

    mask = np.zeros((h, w), dtype=np.uint8)
    i=0
    class_id = None
    for class_id, polygon in polygons:
        if (labels_filter is None) or (class_id in labels_filter):
            x, y = zip(*polygon)
            # Create a Path object from the polygon coordinates
            path_polygon = path.Path(list(zip(x, y)))

            # Generate a mask for the polygon
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            mask_indices = path_polygon.contains_points(points).reshape(h, w)
            # Fill the polygon in the mask
            mask[mask_indices] = 1
            
            i+=1
            if num_samples and (i == num_samples):
                break

    if scaling_pixels:
        scaling_pixels = random.randint(-scaling_pixels, scaling_pixels)
        mask = modify_mask(mask, scaling_pixels)

    return mask, class_id

def create_dataset(image_paths, label_paths, num_samples=None, resize_shape=None, scaling_pixels=None, labels_filter=None):
    images = []
    masks = []
    labels = []

    # list of all the images in the image folder
    images_names = os.listdir(image_paths)
    label_names = [image_name.replace(".jpg", ".txt") for image_name in images_names]
    for img_name, label_name in zip(images_names, label_names):
        if (label_name not in os.listdir(label_paths)):
            images_names.remove(img_name)
            label_names.remove(label_name)
        img_path = os.path.join(image_paths, img_name)
        label_path = os.path.join(label_paths, label_name)

        # read the rgb image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_size = (img.shape[0], img.shape[1])
        polygons = parse_labels(label_path)
        scaled_polygons = scale_polygons(polygons, image_size)
        mask, label = generate_masks(scaled_polygons, image_size, num_samples=num_samples, scaling_pixels=scaling_pixels, labels_filter=labels_filter)

        # resize the image
        if resize_shape is not None:
            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_AREA)

        if len(np.unique(mask)) > 1:
            images.append(img)
            masks.append(mask)
            labels.append(label)

    return np.array(images), np.array(masks), np.array(labels)

def get_images_and_labels_paths(instance_data_root):
    img_path = os.path.join(instance_data_root, "images")
    label_path = os.path.join(instance_data_root, "labels")
    images = []
    masks = []

    # list of all the images in the image folder
    images_names = os.listdir(img_path)
    label_names = [image_name.replace(".jpg", ".txt") for image_name in images_names]
    for img_name, label_name in zip(images_names, label_names):
        if (label_name in os.listdir(label_path)):
            images.append(os.path.join(img_path, img_name))
            masks.append(os.path.join(label_path, label_name))

    return images, masks

def filter_images_and_labels_paths(images, labels, labels_filter):
    filtered_images = []
    filtered_labels = []
    for image, label in zip(images, labels):
        polygons = parse_labels(label)
        if label in labels_filter:
            filtered_images.append(image)
            filtered_labels.append(label)

    return filtered_images, filtered_labels

# mask the images with the masks
def mask_image(image, mask, invert=False):
    if invert:
        return image * (1 - mask)
    else:
        return image * mask


def get_label_mapping(data_yaml):
    with open(data_yaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    label_mapping = {}
    for i, label in enumerate(data['names']):
        label_mapping[i] = label

    return label_mapping

def print_trainable_parameters(model, model_name):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Model: {model_name}")
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_module_kohya_state_dict(
    state_dict, 
    prefix: str, 
    lora_alpha: int,
    dtype: torch.dtype = torch.float32, 
):
    kohya_ss_state_dict = {}
    for peft_key, weight in state_dict.items():
        kohya_key = peft_key.replace("model", prefix, 1)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(lora_alpha).to(dtype)

    return kohya_ss_state_dict

def save_loras(
    unet = None,
    text_encoder = None,
    save_path = "./lora.safetensors",
    config = None,
):  
    lora_alpha = config.lora.alpha
    weights = {}
    metadata = {}

    if config.lora.output_format != 'kohya_ss' and config.lora.output_format != 'peft':
        raise ValueError(f'Invalid output format {config.lora.output_format}')

    if unet:
        state_dict = get_peft_model_state_dict(unet, adapter_name=config.lora.unet_adapter_name)
        if config.lora.output_format == 'kohya_ss':
            state_dict = get_module_kohya_state_dict(state_dict, config.lora.unet_adapter_name, lora_alpha)
        weights.update(state_dict)
    if text_encoder:
        state_dict = get_peft_model_state_dict(text_encoder, adapter_name=config.lora.text_encoder_adapter_name)
        if config.lora.output_format == 'kohya_ss':
            state_dict = get_module_kohya_state_dict(state_dict, config.lora.text_encoder_adapter_name, lora_alpha)
        weights.update(state_dict)
    
    save_file(weights, save_path, metadata)

def save_textual_inversion(
    tokens,
    tokens_ids,
    text_encoder,
    save_path = "./textual_inversion.safetensors",
):  
    embeddings = {}
    for tok, tok_id in zip(tokens, tokens_ids):
        embedding = text_encoder.get_input_embeddings().weight[tok_id]
        embeddings[tok] = embedding.detach().cpu()

    save_file(embeddings, save_path)