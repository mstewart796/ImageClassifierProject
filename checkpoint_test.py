import torch
from train import save_checkpoint

def print_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Print the checkpoint contents
    for key, values in checkpoint.items():
        print(f"{key} {values}")
# arch_test = 'vgg16'
# model = 'test'
# train_dataset = 'test'
# optimizer = 'test'
# epochs = 10
# arch = arch_test
# hidden_units = 100
# save_dir = 'checkpoint.pth'

# save_checkpoint(model, train_dataset, optimizer, epochs, arch, hidden_units, save_dir)
print_checkpoint('checkpoint_test.pth')
