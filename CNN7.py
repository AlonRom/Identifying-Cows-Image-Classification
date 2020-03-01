import torch.nn as nn
import torch.nn.functional as F


class CNN7(nn.Module):

    def __init__(self, num_of_cows, activation_func):
        super(CNN7, self).__init__()

        self.number_of_cows_to_learn = num_of_cows
        self.activation_func = activation_func

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.conv3 = nn.Conv2d(10, 5, 2)
        self.conv4 = nn.Conv2d(5, 10, 3)
        self.conv5 = nn.Conv2d(10, 5, 3)

        # now a few fully connected layers
        self.fc1 = nn.Linear(405, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.number_of_cows_to_learn)

    def forward(self, x):
        x = F.normalize(F.max_pool2d(self.activation_func(self.conv1(x)), 2))
        x = F.normalize(F.max_pool2d(self.activation_func(self.conv2(x)), 2))
        x = F.normalize(self.activation_func(self.conv3(x)))
        x = F.dropout(F.normalize(self.activation_func(self.conv4(x))), 0.3)
        # x = F.normalize(self.activation_func(self.conv4(x)))
        x = F.normalize(self.activation_func(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(F.normalize(self.activation_func(self.fc1(x))), 0.3)
        x = self.activation_func(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
