import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from generator import Generator

# Load pre-trained models
checkpoint = torch.load('cycle_gan_checkpoint_epoch_9.pt', map_location=device)
G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)
G_XtoY.load_state_dict(checkpoint['G_XtoY_state_dict'])
G_YtoX.load_state_dict(checkpoint['G_YtoX_state_dict'])

# Set models to evaluation mode
G_XtoY.eval()
G_YtoX.eval()

# Define the transforms for test data
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load test data
test_dataset_X = datasets.ImageFolder(root='E:\\Sem 4\\Machine learning\\test day', transform=test_transform)
test_dataset_Y = datasets.ImageFolder(root='E:\\Sem 4\\Machine learning\\test night', transform=test_transform)
test_loader_X = DataLoader(test_dataset_X, batch_size=1, shuffle=False)
test_loader_Y = DataLoader(test_dataset_Y, batch_size=1, shuffle=False)

# Directory to save the generated images
save_dir = "./test_results"
os.makedirs(save_dir, exist_ok=True)
