# AI Programming with Python - Image Classifier Project

Image classifier for 102 flower species using PyTorch and transfer learning.

## Usage

### Training
```bash
# Basic training
python train.py data/flowers

# With options
python train.py data/flowers --arch vgg13 --epochs 5 --gpu
```

### Prediction
```bash
# Basic prediction
python predict.py assets/Flowers.png checkpoint.pth

# With flower names
python predict.py assets/Flowers.png checkpoint.pth --category_names cat_to_name.json --top_k 5
```

## Files

- `train.py` - Command line application for training
- `predict.py` - Command line application for predictions
- `Image Classifier Project.ipynb` - Jupyter notebook with full implementation
- `cat_to_name.json` - Mapping of categories to flower names

## Results

After 5 epochs of training:
- Validation accuracy: ~80%

## Requirements

- PyTorch
- torchvision
- PIL
- matplotlib
- numpy
