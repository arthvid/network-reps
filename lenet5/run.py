import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import pdb
from network import LeNet5
from tqdm import tqdm


# Hyperparameters
batch_size = 4
# Prepare the dataset.
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

training_set = torchvision.datasets.MNIST(root='.', download=True, train=True, transform=transform)
validation_set = torchvision.datasets.MNIST(root='.', download=True, train=False, transform=transform)

# Create the dataloader.
train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Instantiate an instance of the network.
model = LeNet5()

# Select a loss function.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters())


number_epochs = 10

for epoch in tqdm(range(number_epochs)):
    running_loss = 0.
    for idx, batch in enumerate(train_loader):
        # Split element of dataloader into data points and labels.
        inputs, labels = batch
        # Zero any accumulated gradients inside optimizer.
        optimizer.zero_grad()
        # Execute forward pass.
        pred = model(inputs)
        # Calculate loss.
        loss = loss_fn(pred, labels)
        # Calculate gradients in backward pass.
        loss.backward()
        # Update gradients.
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
    
        
print()
        
    