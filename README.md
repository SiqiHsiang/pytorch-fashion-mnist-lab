# PyTorch Fashion-MNIST Classifier

A beginner-friendly PyTorch project that builds and trains a simple neural network on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.
It demonstrates how to structure a small ML experiment with a clean src/ pipeline and optional Jupyter exploration.

## Project Structure
```bash
pytorch-fashion-mnist-lab/
├── notebooks/
│   └── pytorch_mnist.ipynb        # Interactive version (for exploration)
├── src/
│   ├── model.py                   # MLP model definition
│   └── train.py                   # Training, evaluation, and saving
├── models/                        # Saved weights (ignored by Git)
├── mni/                           # Virtual environment (ignored)
├── .gitignore
└── README.md
```

## 1. Set and Usage

1. Clone the repository
  ```bash
  git clone https://github.com/YOUR_USERNAME/pytorch-fashion-mnist-lab.git
  cd pytorch-fashion-mnist-lab
  ```

2. (Optional) Create and activate a virtual environment
  ```bash
  python -m venv mni
  source mni/bin/activate     # macOS/Linux
  mni\Scripts\activate        # Windows
  ```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Run the training script
```bash
python -m src.train \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.01 \
  --save models/model.pt
```
The script will:
	•	Download Fashion-MNIST automatically via torchvision
	•	Split a small validation set (10%)
	•	Train an MLP with ReLU + Dropout
	•	Print validation accuracy and F1 after each epoch
	•	Save weights to models/model.pt

## 3. Load trained model
```bash
import torch
from src.model import CustomNetwork

model = CustomNetwork()
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()
```

## Notes
	•	The dataset will be downloaded automatically to ~/.torch.
	•	model.pt and data/ are ignored in .gitignore.
	•	You can also open pytorch_mnist.ipynb for the notebook version.
