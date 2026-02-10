"""
Predict flower name from an image.
Usage: python predict.py /path/to/image checkpoint.pth

Test commands:
    # Simple prediction with example image:
    python predict.py assets/Flowers.png checkpoint.pth

    # Top 5 predictions with flower names:
    python predict.py assets/Flowers.png checkpoint.pth --top_k 5 --category_names cat_to_name.json

    # Test with actual test image:
    python predict.py data/flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json
"""

import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image


# Get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Path to checkpoint')
parser.add_argument('--top_k', type=int, default=1, help='Return top K predictions')
parser.add_argument('--category_names', help='JSON file with category names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

# Rebuild model based on saved architecture
arch = checkpoint.get('arch', 'vgg13')  # Default to vgg13 for old checkpoints

if arch == 'vgg13':
    model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
elif arch == 'vgg16':
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
else:
    raise ValueError(f"Unsupported architecture: {arch}")

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Load saved classifier
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
model.to(device)
model.eval()

# Process image
image = Image.open(args.image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# Make prediction
with torch.no_grad():
    output = model(image)
    ps = torch.exp(output)
    top_p, top_idx = ps.topk(args.top_k, dim=1)

# Convert to lists
probs = top_p.cpu().numpy()[0]
indices = top_idx.cpu().numpy()[0]

# Convert indices to classes
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
classes = [idx_to_class[i] for i in indices]

# Load category names if provided
if args.category_names:
    with open(args.category_names) as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[c] for c in classes]
else:
    names = classes

# Print results
print("\nPredictions:")
for i in range(len(probs)):
    print(f"  {names[i]}: {probs[i]:.3f}")
