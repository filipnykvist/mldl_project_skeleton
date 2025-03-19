import torch
from torch import nn
from models.customnet import CustomNet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import wandb

# Training loop
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Compute prediction and loss
        pred = model(inputs)
        loss = criterion(pred, targets)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log every 100 batches.
        if batch_idx % 100 == 0:
            current_loss = loss.item()
            current = batch_idx * len(inputs)
            print(f"loss: {current_loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")
            wandb.log({"Train Loss": current_loss})
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    wandb.log({"Train Epoch Loss": train_loss, "Train Epoch Accuracy": train_accuracy})
    return train_accuracy


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="tiny-imagenet-customnet", name="My first CustomNet-Training")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)

    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch} -------------------")
        
        # Train for one epoch and store the accuracy.
        train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

        scheduler.step()
        
        # Save the model at the end of training (not overwriting the best one).
        model_path = 'models/last_model.pth'
        torch.save(model.state_dict(), model_path)

        if train_accuracy > best_acc:
            best_acc = train_accuracy
            # Saving the new model.
            model_path = 'models/best_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
            # Log model as a W&B artifact
            artifact = wandb.Artifact('customnet-best-model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
