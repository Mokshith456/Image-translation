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
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define U-Net generator architecture here
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Implement forward pass of the generator
        x = self.encoder(x)
        x = self.decoder(x)
        return x
