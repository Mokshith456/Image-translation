import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from generator import Generator
import os
import numpy as np
from torchvision.utils import save_image
from evaluation_metrics import ssim

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

# Test the model
mse_fake_Y_list = []
mse_fake_X_list = []
ssim_fake_Y_list = []
ssim_fake_X_list = []

with torch.no_grad():
    for i, (batch_X, batch_Y) in enumerate(zip(test_loader_X, test_loader_Y)):
        real_X, _ = batch_X
        real_Y, _ = batch_Y
        real_X = real_X.to(device)
        real_Y = real_Y.to(device)

        # Generate images
        fake_Y = G_XtoY(real_X)
        fake_X = G_YtoX(real_Y)

        # Calculate MSE and SSIM
        mse_fake_Y = torch.nn.functional.mse_loss(fake_Y, real_Y)
        mse_fake_X = torch.nn.functional.mse_loss(fake_X, real_X)
        ssim_fake_Y = ssim(fake_Y, real_Y)
        ssim_fake_X = ssim(fake_X, real_X)

        mse_fake_Y_list.append(mse_fake_Y.item())
        mse_fake_X_list.append(mse_fake_X.item())
        ssim_fake_Y_list.append(ssim_fake_Y)
        ssim_fake_X_list.append(ssim_fake_X)

        # Save generated images
        save_image(fake_Y, os.path.join(save_dir, f"fake_Y_{i}.png"))
        save_image(fake_X, os.path.join(save_dir, f"fake_X_{i}.png"))

    avg_mse_fake_Y = np.mean(mse_fake_Y_list)
    avg_mse_fake_X = np.mean(mse_fake_X_list)
    avg_ssim_fake_Y = np.mean(ssim_fake_Y_list)
    avg_ssim_fake_X = np.mean(ssim_fake_X_list)

print(f"Average MSE (fake_Y): {avg_mse_fake_Y:.4f}, Average MSE (fake_X): {avg_mse_fake_X:.4f}")
print(f"Average SSIM (fake_Y): {avg_ssim_fake_Y:.4f}, Average SSIM (fake_X): {avg_ssim_fake_X:.4f}")

