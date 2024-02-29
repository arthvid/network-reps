import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import pdb
from network import LeNet5


# Prepare the dataset.
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

training_set = torchvision.datasets.MNIST(root='.', download=True, train=True, transform=transform)
validation_set = torchvision.datasets.MNIST(root='.', download=True, train=False, transform=transform)

# Create the dataloader.
train_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Instantiate an instance of the network.
model = LeNet5()

# Select a loss function.
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters)



