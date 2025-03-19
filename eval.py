import torch
from torch import nn
from models.customnet import CustomNet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import wandb

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            pred = model(inputs)
            loss = criterion(pred, targets)

            val_loss += loss.item()
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    
    wandb.log({
        "Evaluation Loss": val_loss,
        "Evaluation Accuracy": val_accuracy
    })
    
    return val_loss, val_accuracy


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="tiny-imagenet-customnet", name="My first CustomNet-Evaluation")

    # Define transformations (same as in train.py)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the validation dataset
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

    # Load the trained model
    model = CustomNet().cuda()
    model.load_state_dict(torch.load('models/best_model.pth'))  # Make sure this matches the file name you saved during training

    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    validate(model, val_loader, criterion)
