import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

# Get the command line instructions
def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    
    This function uses Python's argparse module to created and defined these 
    command line arguments. If the user fails to provide some or all of the 
    arguments, then the default values are used for the missing arguments. 

    Command Line Arguments:
      1. Data Directory as a positional argument
      2. --save_dir: Directory to save checkpoints
      3. --arch: CNN Model architecture to use
      4. --learning_rate: Learning rate for training the model
      5. --hidden_units: Number of hidden units in classifier
      6. --epochs: Number of epochs for training
      7. --gpu: Option to use GPU for training

    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    
    # Add positional argument for data directory
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    
    # Add optional arguments - I have used my best parameters from part 1 as the defaults, EXCEPT FOR LEARNING RATE!!!
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Choose architecture (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.003, # PLEASE test with this value, I spent DAYS stuck debugging when the problem was a LR of 0.01/0.03
                        help='Learning rate (default: 0.003)')
    parser.add_argument('--hidden_units', type=int, default=1024, # I did my tsting mainly with 512 as per the hyperparameters in the instructions
                        help='Number of hidden units (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, # I tested this with 2 epochs because of personal time restrictions
                        help='Number of epochs (default: 10)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    
    return parser.parse_args()

#Data loading and transformations
def load_data(data_dir):
    """
    Create a function that takes the data_dir and loads the images
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    return train_dir, valid_dir, test_dir

def transform_data(data_dir):
  """
  Create a function that uses the  load_data function to load and then transform the data

  Args:
      data_dir : data directory
  """
  train_dir, valid_dir, test_dir = load_data(data_dir)

  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

  valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
  # Load the datasets with ImageFolder
  train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
  test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
  valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

  # Using the image datasets and the trainforms, define the dataloaders
  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
  validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

  return trainloader, testloader, validloader


# Model building
def load_model(arch='vgg16', hidden_units=1024):
  """
  Loads the model

  Args:
      arch (str, optional): Takes the model type. Defaults to "vgg16".
      hidden_units (int, optional): Takes number of hidden units. Defaults to 1024.

  Returns:
      model: returns the model
  """
  model = getattr(models, arch)(weights='DEFAULT')
  # Freeze parameters so we don't backprop through them
  for param in model.parameters():
    param.requires_grad = False
  
  classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)), # I tested with 4096 (the default) and 2048 but got terrible results
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)), # I have chosen 0.2 over the 0.5 default as it gave me poor results
                          ('fc2', nn.Linear(hidden_units, 102)),  # 102 is based on the number of classes in the cat_to_name.json file                            
                          ('output', nn.LogSoftmax(dim=1)) # I tested with more hidden layers but they gave me terrible results
                          ]))
    
  model.classifier = classifier

  return model
  
# Function to train the model
def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device, print_every=5):
    """
    Trains the model.

    Args:
        model: The neural network model to train
        trainloader: DataLoader for training data
        validloader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        epochs: Number of epochs to train
        device: Device to train on (cuda or cpu)
        print_every: Number of steps between printing progress

    Returns:
        model: The trained model
    """
    steps = 0
    running_loss = 0
    running_accuracy = 0

    for epoch in range(epochs):
        model.train()
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            
            running_loss += loss.item()
            running_accuracy += accuracy
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate_model(model, validloader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs} || "
                      f"Training loss: {running_loss/print_every:.2f} || "
                      f"Training accuracy: {running_accuracy/print_every:.2f} || "
                      f"Validation loss: {valid_loss:.2f} || "
                      f"Validation accuracy: {valid_accuracy:.2f}")
                
                running_loss = 0
                running_accuracy = 0
                model.train()
       
    return model

# Function to run validation
def validate_model(model, dataloader, criterion, device):
    """
    Validates the model.

    Args:
        model: The neural network model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda or cpu)

    Returns:
        float: Average validation loss
        float: Average validation accuracy
    """
    model.eval()
    valid_loss = 0
    valid_accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            
            valid_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss / len(dataloader), valid_accuracy / len(dataloader)

# Function to save the checkpoint
def save_checkpoint(model, train_dataset, optimizer, epochs, arch, hidden_units, save_dir='checkpoint.pth'):
    """
    Saves a checkpoint of the trained model.

    Args:
        model (torch.nn.Module): The trained model
        train_dataset (torchvision.datasets.ImageFolder): The training dataset
        optimizer (torch.optim.Optimizer): The optimizer used for training
        epochs (int): The number of epochs the model was trained for
        arch (str): The architecture of the model
        hidden_units (int): The number of hidden units in the classifier
        save_dir (str, optional): Directory to save the checkpoint. Defaults to 'checkpoint.pth'.

    Returns:
        None
    """
    directory = os.path.dirname(save_dir)

    # If directory is not empty, create it if it doesn't exist
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Get the class_to_idx mapping from the training dataset
    class_to_idx = train_dataset.class_to_idx

    # Create the checkpoint dictionary
    checkpoint = {
        'arch': arch,
        'input_size': 25088, 
        'hidden_units': hidden_units,
        'output_size': 102,  
        'class_to_idx': class_to_idx,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'state_dict': model.state_dict()
    }

    # Save the checkpoint
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved to {save_dir}")
    print(f'arch: {arch}')

def main():
    # Get command line arguments
    args = get_input_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # Load and transform data
    trainloader, testloader, validloader = transform_data(args.data_dir)

    # Load model
    model = load_model(args.arch, args.hidden_units)
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    model = train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)

    # Test the model
    test_loss, test_accuracy = validate_model(model, testloader, criterion, device)
    print(f"Final Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")

    # Save the checkpoint
    save_checkpoint(model, trainloader.dataset, optimizer, args.epochs, args.arch, args.hidden_units, args.save_dir)

    print("Training complete!")

if __name__ == "__main__":
    main()
