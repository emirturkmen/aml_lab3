import torch
from torch import nn
from models.customnet import CustomNet
from dataset.dataloader import get_dataloaders
from train import validate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=32)

    model = CustomNet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()
    validate(model, val_loader, criterion, device)

if __name__ == "__main__":
    main()