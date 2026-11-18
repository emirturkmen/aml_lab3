import os
import torch
from torch import nn
import wandb

from models.customnet import CustomNet
from dataset.dataloader import get_dataloaders

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_acc:.2f}%')
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            val_loss += loss.item()
            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_acc:.2f}%')
    return val_loss, val_acc

def main():
    # 1. WandB başlat[cite: 11]
    wandb.init(project="aml-lab3", config={
        "learning_rate": 0.001,
        "momentum": 0.9,
        "batch_size": 32,
        "epochs": 10
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(batch_size=config.batch_size)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Metrikleri WandB'ye gönder[cite: 9, 11]
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # En iyi modeli kaydet[cite: 15]
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()

if __name__ == "__main__":
    main()