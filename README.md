# AI Programming with Python Project
## Overview

This project focuses on developing an AI application capable of classifying different species of flowers. Using a deep learning model, the classifier has been trained on a dataset containing 102 flower categories. The trained model can predict flower types from given images and can be extended to classify other labeled datasets.

This project is part of the [Udacity AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).

## Features

- Load and preprocess datasets for training, validation, and testing.
- Train a neural network using a pre-trained model (transfer learning).
- Test the model's accuracy on unseen data.
- Predict flower species from an input image.
- Deploy a command-line interface for predictions.

## Dataset

The project uses the [Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which contains 102 categories of flowers.

## Getting Started

### Prerequisites

Ensure the following libraries are installed:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/image-classifier-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd image-classifier-project
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow

1. **Data Loading and Preprocessing**
   - Training data: Includes random scaling, cropping, and flipping for augmentation.
   - Validation and testing data: Only resized and cropped to maintain consistency.

2. **Model Training**
   - Utilizes a pre-trained model from PyTorch (`torchvision.models`).
   - Adds a custom classifier to the pre-trained model.
   - Fine-tunes the model with training data.

3. **Testing and Validation**
   - Evaluates model performance on unseen data.

4. **Prediction**
   - Accepts an image path and returns the predicted flower category along with the probability.

## Usage

### Train the Model

Run the notebook or a training script to load the dataset and train the model:
```bash
python train.py --data_dir ./flowers --save_dir ./checkpoint.pth --arch vgg16 --epochs 10
```

### Make Predictions

Use the trained model to predict the class of a flower:
```bash
python predict.py --image_path ./flowers/test/1/image_06743.jpg --checkpoint ./checkpoint.pth
```

### Command-Line Arguments

- `--data_dir`: Directory containing the dataset.
- `--save_dir`: Path to save the trained model checkpoint.
- `--arch`: Pre-trained model architecture (e.g., `vgg16` or `resnet50`).
- `--image_path`: Path to the input image.
- `--top_k`: Number of top predictions to display.

## Results

The model achieves high accuracy on the test dataset and can reliably predict the category of various flowers. See the notebook for more details.

## Future Work

- Extend the model to other datasets.
- Deploy the model in a mobile application.
- Improve the model with additional training data or architecture tuning.

## Acknowledgments

- Dataset: [Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- Udacity AI Programming with Python Nanodegree.
