import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

sys.path.append('classes')
from NeuralNetwork_baseline import Net_baseline
from NeuralNetwork_defense1 import Net_defense1
from NeuralNetwork_defense3 import Net_defense3


# ================================================== FUNCTIONS DECLARATION ================================================== #


def generate_adversarial_example(model, data, target, epsilon):
    """
    Generate adversarial examples using FGSM.
    
    Args:
        model (torch.nn.Module): The neural network model.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Ground truth labels.
        epsilon (float): Perturbation strength.

    Returns:
        torch.Tensor: Adversarial examples.
    """
    data.requires_grad = True  # Enable gradient tracking for the input

    # Forward pass to compute the loss
    output = model(data)
    loss = F.nll_loss(output, target)

    # Backward pass to compute gradients of the loss w.r.t. inputs
    model.zero_grad()
    loss.backward()

    # Generate adversarial perturbation
    data_grad = data.grad.data
    perturbation = epsilon * data_grad.sign()
    
    # Add the perturbation to the input
    adversarial_data = data + perturbation
    
    # Clip the perturbed data to keep it in valid range [0, 1]
    adversarial_data = torch.clamp(adversarial_data, 0, 1)
    
    return adversarial_data


def train(model, device, train_loader, optimizer):
    """
    Train the model with default configuration.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to run on.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer for training.
    """
    model.train()

    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
            
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    return(model, running_loss)


clipping = ""
def train_with_clipping_gradients(model, device, train_loader, optimizer):
    """
    Train the model with clipping gradients.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to run on.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer for training.
    """
    model.train()

    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        
        # Apply gradient clipping here
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    clipping = "-clipping"
    return (model, running_loss)


adversarial = ""
def train_with_adversarial_examples(model, device, train_loader, optimizer):
    """
    Train the model with adversarial examples.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to run on.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        epsilon (float): Perturbation strength for adversarial examples.
    """
    epsilon = 0.1  # Perturbation strength for FGSM
    
    model.train()
    
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # ================== Clean Examples ==================
        optimizer.zero_grad()  # Clear gradients
        output_clean = model(data)  # Forward pass
        loss_clean = F.nll_loss(output_clean, target)  # Compute loss for clean data

        # ================== Adversarial Examples ==================
        data_adv = generate_adversarial_example(model, data, target, epsilon)  # Generate adversarial examples
        output_adv = model(data_adv)  # Forward pass on adversarial examples
        loss_adv = F.nll_loss(output_adv, target)  # Compute loss for adversarial examples

        # ================== Combine Losses ==================
        loss = loss_clean + loss_adv  # Combine clean and adversarial losses
        loss.backward()  # Backward pass (compute gradients)

        optimizer.step()  # Update model parameters

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    adversarial = "-adversarial"
    return (model, running_loss)


# ================================================= PARAMETERS ============================================================== #

Net = Net_defense3

# ================================================= BEGINNING OF EXECUTION ================================================== #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Net(device = device)

transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = torchvision.datasets.FashionMNIST(root = "data/",
                                  train = True,
                                  transform = transform,
                                  target_transform = None,
                                  download = True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

num_epochs = 10
for epoch in range(num_epochs):
    #model, running_loss = train(model, device, train_loader, optimizer)
    #model, running_loss = train_with_clipping_gradients(model, device, train_loader, optimizer)
    model, running_loss = train_with_adversarial_examples(model, device, train_loader, optimizer)

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

torch.save(model.state_dict(), "trained_models/lenet_fashion_mnist_model"+Net.__str__()+clipping+adversarial+".pth")