import torch.nn as nn
import torch.nn.functional as F


class CNN3(nn.Module):

    def __init__(self, num_of_cows, activation_func):
        super(CNN3, self).__init__()

        self.training = True

        self.number_of_cows_to_learn = num_of_cows
        self.activation_func = activation_func

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 10, 2)
        self.conv4 = nn.Conv2d(10, 20, 1)
        self.conv5 = nn.Conv2d(20, 10, 1)

        # now a few fully connected layers
        self.fc1 = nn.Linear(10 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.number_of_cows_to_learn)

    # F.relu
    def forward(self, x):
        x = F.max_pool2d(self.activation_func(self.conv1(x)), (2, 2))
        x = F.normalize(self.activation_func(self.conv2(x)))
        x = F.dropout(self.activation_func(self.conv3(x)), training=self.training)
        x = F.max_pool2d(x, 2)
        x = F.normalize(self.activation_func(self.conv4(x)))
        x = F.dropout(self.activation_func(self.conv5(x)), training=self.training)

        x = x.view(-1, self.num_flat_features(x))
        x = self.activation_func(self.fc1(x))
        x = F.normalize(self.activation_func(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def is_training(self, training=False):
        self.training = training

