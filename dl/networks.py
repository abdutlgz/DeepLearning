import torch
import torch.nn as nn 

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
            self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)  # 5x5 is the size after two convs and poolings
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        #conv+relu+maxpool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        #conv+relu+maxpool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        #flatten tensor
        x = x.view(-1, 16 * 5 * 5) 
        
        # Fully connected layers + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        #output
        x = self.fc3(x)
        return x
    
class Dense(nn.Module):
    """
    Fully Connected feed forward network.

    `depth` is the total number of linear layers.
    for this I used pytorch documentation
    """
    def __init__(self, in_dim, out_dim, width, depth):
        super(Dense, self).__init__()
        #starting the same way as Shallow
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=width, bias=True),
            nn.ReLU()
        )
        #iterating through depth
        for i in range(depth):
            self.layers.add_module(f'hidden_{i}', nn.Linear(in_features=width, out_features=width, bias=True))
            self.layers.add_module(f'activation_{i}', nn.ReLU())

        #output layer
        self.layers.add_module('output', nn.Linear(in_features=width, out_features=out_dim, bias=False))

    def forward(self, x):
        return self.layers(x)

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth):
        #nn.Module parent
        super(Dense, self).__init__()
        #list of all layers
        layers = [
            nn.Linear(in_features=in_dim, out_features=width),
            nn.ReLU()
        ]
        #appending all layers till depth-2
        for _ in range(depth - 2):
            layers.append(nn.Linear(in_features=width, out_features=width))
            layers.append(nn.ReLU())
        #output layer
        layers.append(nn.Linear(in_features=width, out_features=out_dim))
        #putting all layers to Sequential, this way you mentioned in class
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    
class Shallow(nn.Module):
    """
    Shallow feed forward network.
    """
    def __init__(self, in_dim, out_dim, width):
        super(Shallow, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=width, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=width, out_features=out_dim, bias=False),
        )

    def forward(self, x):
        return self.layers(x)
