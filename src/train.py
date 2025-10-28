# src/train.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from .model import CustomNetwork

def get_loaders(batch_size: int = 128):
    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root="~/.torch", train=True, download=True, transform=tfm)
    test_ds  = datasets.FashionMNIST(root="~/.torch", train=False, download=True, transform=tfm)

    # build a small validation split from the training set (10%)
    n_total = len(train_ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return acc, f1

def main():
    ap = argparse.ArgumentParser(description="Train MLP on Fashion-MNIST")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)   # match notebook's SGD lr ~ .01
    ap.add_argument("--save", type=str, default="models/model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl = get_loaders(batch_size=args.batch_size)

    model = CustomNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        acc_val, f1_val = evaluate(model, val_dl, device)
        print(f"epoch {epoch:02d} | loss {loss:.4f} | val_acc {acc_val:.3f} | val_f1 {f1_val:.3f}")

    acc_test, f1_test = evaluate(model, test_dl, device)
    print(f"test_acc {acc_test:.3f} | test_f1 {f1_test:.3f}")

    # Save weights
    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    main()
