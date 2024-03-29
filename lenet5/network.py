import torch 
import torch.nn as nn 
import torchvision 


class LeNet5(nn.Module):

    def __init__(self):
       super().__init__()
       
       self.tanh = nn.Tanh()
       self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1)
       self.avgpool1 = nn.AvgPool2d(kernel_size=(2,2))
       self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1)
       self.avgpool2 = nn.AvgPool2d(kernel_size=(2,2))
       self.fc1 = nn.Linear(in_features=256, out_features=64)
       self.fc2 = nn.Linear(in_features=64, out_features=10)
       self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.tanh(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        logits = self.tanh(x)
        probabilities = self.softmax(logits)

        return probabilities





