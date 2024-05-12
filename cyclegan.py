import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import os
from torchvision.utils import save_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming Generator and Discriminator classes are defined in generator.py and discriminator.py respectively
from generator import Generator
from discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self, G_XtoY, G_YtoX, D_X, D_Y):
        super(CycleGAN, self).__init__()
        self.G_XtoY = G_XtoY
        self.G_YtoX = G_YtoX
        self.D_X = D_X
        self.D_Y = D_Y

    def forward(self, real_X, real_Y):
        fake_Y = self.G_XtoY(real_X)
        cycle_X = self.G_YtoX(fake_Y)
        fake_X = self.G_YtoX(real_Y)
        cycle_Y = self.G_XtoY(fake_X)

        return fake_Y, cycle_X, fake_X, cycle_Y

torch.cuda.empty_cache()

# Define hyperparameters
batch_size = 2
lr = 0.0002
epochs = 10

# Initialize networks
G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)
D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)

# Initialize CycleGAN model
cycle_gan = CycleGAN(G_XtoY, G_YtoX, D_X, D_Y).to(device)

# Define loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Define optimizers
optimizer_G = optim.Adam(itertools.chain(G_XtoY.parameters(), G_YtoX.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_X = optim.Adam(D_X.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=lr, betas=(0.5, 0.999))

# Data loading
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset_X = datasets.ImageFolder(root='E:/Sem 4/Machine learning/day_time', transform=transform)
dataset_Y = datasets.ImageFolder(root='E:/Sem 4/Machine learning/night_time', transform=transform)
dataloader_X = DataLoader(dataset_X, batch_size=batch_size, shuffle=True)
dataloader_Y = DataLoader(dataset_Y, batch_size=batch_size, shuffle=True)

# Evaluation metrics
def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)

def ssim(img1, img2):
    img1 = img1.cpu().detach().numpy() if isinstance(img1, torch.Tensor) else img1
    img2 = img2.cpu().detach().numpy() if isinstance(img2, torch.Tensor) else img2
    img1 = img1 / 255.0 if img1.max() > 1 else img1
    img2 = img2 / 255.0 if img2.max() > 1 else img2
    ssim_values = []
    for i in range(img1.shape[0]):
        current_img1 = img1[i]
        current_img2 = img2[i]
        min_dim = min(current_img1.shape[:2])
        win_size = min(7, min_dim - (min_dim % 2 - 1))
        ssim_value = ssim_skimage(current_img1, current_img2, win_size=win_size, multichannel=True, data_range=img1.max() - img1.min())
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

# Directory to save the generated images
save_dir = "./generated_images"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(epochs):
    # Iterate through dataset batches
    for batch_idx, (real_X, real_Y) in enumerate(zip(dataloader_X, dataloader_Y)):
        # Extract the data tensor from the list
        real_X, real_Y = real_X[0].to(device), real_Y[0].to(device)

        # Train Generator
        optimizer_G.zero_grad()

        fake_Y, cycle_X, fake_X, cycle_Y = cycle_gan(real_X, real_Y)

        # Calculate Generator adversarial loss
        loss_GAN_X = criterion_GAN(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
        loss_GAN_Y = criterion_GAN(D_X(fake_X), torch.ones_like(D_X(fake_X)))

        # Calculate cycle consistency loss
        loss_cycle_X = criterion_cycle(cycle_X, real_X)
        loss_cycle_Y = criterion_cycle(cycle_Y, real_Y)

        # Calculate identity loss
        identity_X = G_YtoX(real_X)
        identity_Y = G_XtoY(real_Y)
        loss_identity_X = criterion_identity(identity_X, real_X)
        loss_identity_Y = criterion_identity(identity_Y, real_Y)

        # Total Generator loss
        loss_G = loss_GAN_X + loss_GAN_Y + loss_cycle_X + loss_cycle_Y + loss_identity_X + loss_identity_Y

        # Backpropagation
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator X
        optimizer_D_X.zero_grad()

        loss_D_X_real = criterion_GAN(D_X(real_X), torch.ones_like(D_X(real_X)))
        loss_D_X_fake = criterion_GAN(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X.detach())))

        loss_D_X = 0.5 * (loss_D_X_real + loss_D_X_fake)

        loss_D_X.backward()
        optimizer_D_X.step()

        # Train Discriminator Y
        optimizer_D_Y.zero_grad()

        loss_D_Y_real = criterion_GAN(D_Y(real_Y), torch.ones_like(D_Y(real_Y)))
        loss_D_Y_fake = criterion_GAN(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y.detach())))

        loss_D_Y = 0.5 * (loss_D_Y_real + loss_D_Y_fake)

        loss_D_Y.backward()
        optimizer_D_Y.step()

        # Print losses
        if batch_idx % 64 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], Batch Step [{batch_idx}/{min(len(dataloader_X), len(dataloader_Y))}], "
                f"Loss G: {loss_G.item():.4f}, Loss D_X: {loss_D_X.item():.4f}, Loss D_Y: {loss_D_Y.item():.4f}, "
                f"Cycle Consistency Loss X: {loss_cycle_X.item():.4f}, Cycle Consistency Loss Y: {loss_cycle_Y.item():.4f}, "
                f"Identity Loss X: {loss_identity_X.item():.4f}, Identity Loss Y: {loss_identity_Y.item():.4f}"
            )
        # Save the generated images individually
        save_image(fake_Y, os.path.join(save_dir, f"fake_Y_epoch{epoch}_batch{batch_idx}.png"))
        save_image(fake_X, os.path.join(save_dir, f"fake_X_epoch{epoch}_batch{batch_idx}.png"))
        save_image(cycle_X, os.path.join(save_dir, f"cycle_X_epoch{epoch}_batch{batch_idx}.png"))
        save_image(cycle_Y, os.path.join(save_dir, f"cycle_Y_epoch{epoch}_batch{batch_idx}.png"))

    # Evaluation (moved outside the batch loop)
    with torch.no_grad():
        # Sample a batch from dataloader_X
        data_X = next(iter(dataloader_X))
        sample_real_X = data_X[0].to(device)  # Assuming data_X is a tuple where the first element is the input batch

        # Sample a batch from dataloader_Y
        data_Y = next(iter(dataloader_Y))
        sample_real_Y = data_Y[0].to(device)  # Assuming data_Y is a tuple where the first element is the input batch

        if len(sample_real_Y.shape) > 3:  # Check for batch dimension
            sample_real_Y = sample_real_Y.squeeze(0)  # Remove batch dimension
            sample_real_X = sample_real_X.squeeze(0)  # Remove batch dimension for consistency

        # Perform evaluation using the sampled batches
        fake_Y, cycle_X, fake_X, cycle_Y = cycle_gan(sample_real_X, sample_real_Y)

        # Calculate evaluation metrics
        # For example, Mean Squared Error (MSE) and Structural Similarity Index (SSIM)
        mse_fake_Y = mse(fake_Y, sample_real_Y)
        mse_fake_X = mse(fake_X, sample_real_X)
        ssim_fake_Y = ssim(fake_Y, sample_real_Y)
        ssim_fake_X = ssim(fake_X, sample_real_X)

        # Print evaluation metrics
        print(f"Evaluation Metrics - Epoch [{epoch}/{epochs}]:")
        print(f"  MSE (fake_Y): {mse_fake_Y:.4f}, MSE (fake_X): {mse_fake_X:.4f}")
        print(f"  SSIM (fake_Y): {ssim_fake_Y:.4f}, SSIM (fake_X): {ssim_fake_X:.4f}")

    # Save models periodically
    if epoch % 9 == 0:
        torch.save({
            'G_XtoY_state_dict': G_XtoY.state_dict(),
            'G_YtoX_state_dict': G_YtoX.state_dict(),
            'D_X_state_dict': D_X.state_dict(),
            'D_Y_state_dict': D_Y.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_X_state_dict': optimizer_D_X.state_dict(),
            'optimizer_D_Y_state_dict': optimizer_D_Y.state_dict(),
            'epoch': epoch
        }, f"cycle_gan_checkpoint_epoch_{epoch}.pt")

