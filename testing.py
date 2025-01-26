import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

sys.path.append('classes')
from NeuralNetwork import Net


# ================================================= BEGINNING OF EXECUTION ================================================== #


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Net(device = device)

model_path= "models/lenet_fashion_mnist_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.eval()

criterion = nn.NLLLoss()

# Load TEST DATASET
transform = transforms.Compose([transforms.ToTensor(),])
test_dataset = torchvision.datasets.FashionMNIST(root = "data/",
                                  train = False,
                                  transform = transform,
                                  target_transform = None,
                                  download = True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False) 

with torch.no_grad():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        test_loss += criterion(output, target).item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"\nTest Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")