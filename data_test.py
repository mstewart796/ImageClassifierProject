import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

"""
I created this file to help with debugging. I was stuck with training not going over 20% and in many cases stuck at 10%.
I wanted to investigate my transformations as I tested many, many other things. Multiple models, multiple layers/hidden units,
different dropout rates and a LR from 0.01 to 0.03. I finally decided to start from scratch and when I accidentally set the 
LR to 0.003 I resolved all the problems
"""

def test_data_transforms(data_dir):
    """
    Test the data transformations and provide detailed information about the dataset.
    
    Args:
    data_dir (str): Path to the data directory containing 'train', 'valid', and 'test' subdirectories.
    
    Returns:
    None (prints information and saves sample images)
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Print dataset information
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(valid_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Number of classes: {len(train_data.classes)}")
    print(f"Class to idx mapping: {train_data.class_to_idx}")

    # Check shapes
    print(f"\nSample training image shape: {next(iter(trainloader))[0][0].shape}")
    print(f"Sample training label: {next(iter(trainloader))[1][0].item()}")

    # Visualize some transformed images
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Get random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    plt.figure(figsize=(10, 5))
    imshow(utils.make_grid(images[:4]))
    plt.title('Sample Transformed Training Images')
    plt.savefig('sample_transformed_images.png')
    print("\nSample transformed images saved as 'sample_transformed_images.png'")

    # Print some statistics
    print("\nDataset Statistics:")
    for loader, name in [(trainloader, "Training"), (validloader, "Validation"), (testloader, "Test")]:
        total = 0
        class_counts = {i: 0 for i in range(len(train_data.classes))}
        for _, labels in loader:
            for label in labels:
                class_counts[label.item()] += 1
                total += 1
        print(f"\n{name} set:")
        print(f"  Total images: {total}")
        for class_idx, count in class_counts.items():
            print(f"  Class {class_idx} ({train_data.classes[class_idx]}): {count} images ({count/total*100:.2f}%)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'flowers'  # default directory
    test_data_transforms(data_dir)
