"""
Train a neural network on flower images.
Usage: python train.py data_directory

Test commands:
    # Quick test (1 epoch, ~10-15 min):
    python train.py data/flowers --epochs 1

    # With VGG16 architecture:
    python train.py data/flowers --arch vgg16 --epochs 1

    # Full training (5 epochs, ~50-75 min):
    python train.py data/flowers --epochs 5 --learning_rate 0.001 --hidden_units 512

    # With custom save directory:
    python train.py data/flowers --epochs 3 --save_dir saved_models
"""

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
import os


# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='Path to dataset')
parser.add_argument('--save_dir', default='.', help='Directory to save checkpoint')
parser.add_argument('--arch', default='vgg13', choices=['vgg13', 'vgg16'], help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and transform data
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_data = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Build model based on architecture
if args.arch == 'vgg13':
    model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    input_size = 25088
elif args.arch == 'vgg16':
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    input_size = 25088

print(f"Using architecture: {args.arch}")

# Freeze pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# Build new classifier
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier
model.to(device)

# Setup training
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train the network
print(f"\nTraining for {args.epochs} epochs...")
for epoch in range(args.epochs):
    train_loss = 0
    model.train()

    # Training loop
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    valid_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            valid_loss += criterion(output, labels).item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_class = ps.topk(1, dim=1)[1]
            equals = top_class == labels.view(*top_class.shape)
            accuracy += equals.type(torch.FloatTensor).mean().item()

    # Print progress
    print(f"Epoch {epoch+1}/{args.epochs} - "
          f"Train loss: {train_loss/len(trainloader):.3f} - "
          f"Valid loss: {valid_loss/len(validloader):.3f} - "
          f"Valid accuracy: {accuracy/len(validloader):.3f}")

# Save checkpoint
checkpoint = {
    'arch': args.arch,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'hidden_units': args.hidden_units
}

os.makedirs(args.save_dir, exist_ok=True)
checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
torch.save(checkpoint, checkpoint_path)
print(f"\nCheckpoint saved to {checkpoint_path}")
