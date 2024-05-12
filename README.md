# CycleGAN for Image Translation

This repository contains code for training and testing a CycleGAN model for image translation tasks. CycleGAN is a type of generative adversarial network (GAN) that learns to translate images from one domain to another without paired data.

## Overview

The CycleGAN model comprises two generators (`G_XtoY` and `G_YtoX`) and two discriminators (`D_X` and `D_Y`). The generators are responsible for transforming images from one domain to another, while the discriminators aim to distinguish between real and generated images.

The training process involves optimizing the following loss functions:

1. **Adversarial Loss**: This loss encourages the generators to produce realistic-looking images that can fool the discriminators.
2. **Cycle Consistency Loss**: This loss ensures that the generated images can be transformed back to their original domain, preserving essential characteristics.
3. **Identity Loss**: This loss encourages the generators to preserve the input image when translating between domains.

During training, the generators and discriminators are optimized in an adversarial manner, with the generators aiming to produce realistic images and the discriminators trying to distinguish between real and generated images.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-image
- numpy

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your_username/CycleGAN-Image-Translation.git
cd CycleGAN-Image-Translation
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train.py
```

4. Test the model:

```bash
python test.py
```

## Directory Structure

- `data/`: Directory to store training and testing data.
- `models/`: Directory to save trained models.
- `results/`: Directory to save generated images during testing.
- `train.py`: Script for training the CycleGAN model.
- `test.py`: Script for testing the trained model.
- `generator.py`: Contains the Generator class.
- `discriminator.py`: Contains the Discriminator class.
- `utils.py`: Utility functions.
- `evaluation_metrics.py`: Functions for evaluating the model.
- `README.md`: This README file.

## Acknowledgements

- This implementation is inspired by the original CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).
- Parts of the code are adapted from [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu et al.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
