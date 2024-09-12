import argparse
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

# Get the command line instructions
def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the `predict.py` program from a terminal window.
    
    This function uses Python's argparse module to create and define these 
    command line arguments. If the user fails to provide some or all of the 
    arguments, then the default values are used for the missing arguments.

    Command Line Arguments:
      1. Image path as a positional argument
      2. Checkpoint path as a positional argument
      3. --top_k: Return the top K most likely classes
      4. --category_names: Use a mapping of categories to real names
      5. --gpu: Option to use GPU for inference

    Returns:
     parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")

    # Add positional arguments for image path and checkpoint path
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint_path', type=str, help='Path to the saved checkpoint')

    # Add optional arguments
    parser.add_argument('--top_k', type=int, default=1,
                        help='Return the top K most likely classes (default: 1)')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to a JSON file mapping categories to real names (default: cat_to_name.json)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')

    return parser.parse_args()

# Create function to map the labels
def label_mapping(cat_to_name):
    """
    Maps category labels to real names using a JSON file.    
 
    Args:
        cat_to_name (str): Path to the JSON file that maps categories to names.

    Returns:
        dict: A dictionary mapping category labels to real names.
    """
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

# Create the function to load the checkpoint
def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the saved model architecture dynamically.

    Args:
        filepath (str): The path to the checkpoint file.

    Returns:
        model (torch.nn.Module): The rebuilt model.
        class_to_idx (dict): Mapping of class indices to class labels.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
    """
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Extract the model architecture from the checkpoint
    arch = checkpoint['arch']
    
    # Dynamically load the model by using the arch string
    model = getattr(models, arch)(weights='DEFAULT')
    
    # Rebuild the classifier with the saved parameters
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    # Load the model's state_dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the class_to_idx mapping
    class_to_idx = checkpoint['class_to_idx']
    
    # Load the optimizer state_dict
    optimizer = torch.optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, class_to_idx, optimizer

# Create the function to process the image
def process_image(image):
    """
    Preprocesses a PIL image for use in a PyTorch model. The function performs
    resizing, cropping, normalization, and transforms the image into a PyTorch
    tensor, ready for model inference.

    Steps:
      1. Opens the image using PIL.
      2. Resizes the image where the shortest side is 256 pixels.
      3. Crops the center 224x224 portion of the image.
      4. Converts the image pixel values to a range of [0, 1].
      5. Normalizes the image using predefined mean and standard deviation values.
      6. Reorders the image dimensions to match PyTorch's expected format.
      7. Converts the NumPy array into a PyTorch tensor.

    Args:
        image (str): Path to the image file.

    Returns:
        torch.Tensor: The preprocessed image ready for input to a PyTorch model.
    """
    
    # Step 1: Open the image using PIL
    pil_image = Image.open(image)
    
    # Step 2: Resize the image, ensuring the shortest side is 256 pixels
    pil_image.thumbnail((256, 256))  # Resizes the image while maintaining aspect ratio

    # Step 3: Crop the center 224x224 portion of the image
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Step 4: Convert pixel values to the range [0, 1] by dividing by 255
    np_image = np.array(pil_image) / 255.0

    # Step 5: Normalize the image using mean and standard deviation values
    means = [0.485, 0.456, 0.406]  # Predefined mean values for RGB channels
    stds = [0.229, 0.224, 0.225]   # Predefined standard deviation values for RGB channels
    np_image = (np_image - means) / stds

    # Step 6: Reorder dimensions from (height, width, channels) to (channels, height, width)
    np_image = np_image.transpose((2, 0, 1))

    # Step 7: Convert the NumPy array into a PyTorch tensor and ensure it's a float type
    tensor_image = torch.from_numpy(np_image).float()
    
    return tensor_image

# Prediction function
def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an input image using a trained deep learning model.

    Args:
        image_path (str): Path to the input image to classify.
        model (torch.nn.Module): The trained model to use for prediction.
        topk (int, optional): Number of top predicted classes to return. Defaults to 5.

    Returns:
        torch.Tensor: The top K probabilities.
        torch.Tensor: The corresponding top K class indices.
    """
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension for model input
    
    # Move the image to the correct device (CPU or GPU)
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Ensure the model is in evaluation mode (disables dropout, etc.)
    model.eval()
    
    with torch.no_grad():  # Disable gradient calculation for prediction
        logps = model(image)
        ps = torch.exp(logps)  # Convert log probabilities to probabilities
        top_p, top_class = ps.topk(topk, dim=1)  # Get top K probabilities and class indices
    
    return top_p, top_class

def main():
    # Get input arguments
    args = get_input_args()
    
    # Load the checkpoint and rebuild the model
    model, class_to_idx, _ = load_checkpoint(args.checkpoint_path)
    
    # Move the model to the GPU if requested and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Make predictions
    top_p, top_class = predict(args.image_path, model, args.top_k)
    
    # Invert class_to_idx to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Map class indices to actual class names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class_labels = [cat_to_name[idx_to_class[i]] for i in top_class.cpu().numpy().squeeze()]
    else:
        top_class_labels = [idx_to_class[i] for i in top_class.cpu().numpy().squeeze()]
    
    # Print the results
    print(f"Top {args.top_k} Predictions:")
    for i in range(len(top_class_labels)):
        print(f"Class: {top_class_labels[i]}, Probability: {top_p.cpu().numpy().squeeze()[i]}")

if __name__ == '__main__':
    main()
