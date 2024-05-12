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
