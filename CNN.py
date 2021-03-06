import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, num_of_cows, activation_func):
        super(CNN, self).__init__()

        self.number_of_cows_to_learn = num_of_cows
        self.activation_func = activation_func

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # now a few fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.number_of_cows_to_learn)

    # F.relu
    def forward(self, x):
        x = F.max_pool2d(self.activation_func(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.activation_func(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
