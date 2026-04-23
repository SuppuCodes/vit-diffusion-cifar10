# Comparative Analysis of CNN, Vision Transformer, and Diffusion Models on CIFAR-10

## Author

Supraja Katta  
Western Illinois University  
Independent Researcher  
supraja.professional.mail@gmail.com

---

## Project Overview

This project presents a comparative study of three major deep learning paradigms:

- Convolutional Neural Networks (CNN)
- Vision Transformers (ViT)
- Diffusion Models

All models are implemented using PyTorch and trained on the CIFAR-10 dataset under CPU-only constraints. The goal is to evaluate classification performance, generative capability, and practical limitations of modern architectures in low-compute environments.

---

## Objectives

- Compare CNN and Vision Transformer performance on CIFAR-10
- Implement a diffusion model for image generation
- Analyze training behavior under limited computational resources
- Understand trade-offs between traditional and modern architectures

---

## Models Implemented

### CNN

- Convolutional layers with fully connected classifier
- Cross-entropy loss
- Strong inductive bias for image data

### Vision Transformer

- Patch embedding
- Transformer encoder layers
- Positional embeddings

### Diffusion Model

- UNet-based architecture
- Noise prediction training
- Linear beta schedule

---

## Results

### Classification Accuracy

| Model | Accuracy |
| ----- | -------- |
| CNN   | 66.86%   |
| ViT   | 49.81%   |

---

### Training Loss

CNN Training Loss  
![CNN Loss](cnn_loss.png)

Vision Transformer Training Loss  
![ViT Loss](vit_loss.png)

Diffusion Training Loss  
![Diffusion Loss](diffusion_loss.png)

---

### Generated Samples

![Generated Images](diffusion_samples.png)

Note: Generated images remain noisy due to limited training steps and CPU-only training.

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/SuppuCodes/vit-diffusion-cifar10.git
cd vit-diffusion-cifar10
```

### 2.Install Dependencies

pip install -r requirements.txt

### 3.Train Models

-- CNN

python -m train.train_cnn

-- Vision Transformer

python -m train.train_vit

-- Diffusion Model

python -m train.train_diffusion

## References

-- AlexNet
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
-- ResNet
https://arxiv.org/pdf/1512.03385.pdf
-- Vision Transformer
https://arxiv.org/pdf/2010.11929.pdf
-- Diffusion Models (DDPM)
https://arxiv.org/pdf/2006.11239.pdf

## Key Insights

-- CNN performs best due to strong spatial inductive bias
-- ViT underperforms in low-data, low-compute settings
-- Diffusion models require significantly more training for meaningful outputs
-- Training modern architectures on CPU reveals real-world constraints often ignored in research

## Future Work

-- Train models on GPU for better performance
-- Improve diffusion sampling quality
-- Experiment with hybrid CNN + Transformer architectures
-- Increase training epochs and dataset size

## Research Paper

-- A full IEEE-style paper based on this work is included in this project.
