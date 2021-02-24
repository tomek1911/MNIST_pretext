import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.3)     
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(20*6*6, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 20*6*6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return x


    
    
