import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

sys.path.append('classes')
from NeuralNetwork import Net

# Load the train dataset
transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = torchvision.datasets.FashionMNIST(root = "data/",
                                  train = True,
                                  transform = transform,
                                  target_transform = None,
                                  download = True)

print(f"Number of training samples: {len(train_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Net(device = device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target to the selected device
        data, target = data.to(device), target.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")


# Save the trained model
torch.save(model.state_dict(), "models/lenet_fashion_mnist_model.pth")
print("Model saved as models/lenet_fashion_mnist_model.pth!")