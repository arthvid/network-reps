import torch
from network import LeNet5


x = torch.rand((3,32,32))
network = LeNet5()

y = network(x)

print(x)
print(y)
