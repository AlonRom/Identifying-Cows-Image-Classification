

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from CNN import CNN
from CNN2 import CNN2
from CNN3 import CNN3
from CNN4 import CNN4
from CNN5 import CNN5
from CNN6 import CNN6
from CNN8 import CNN8



def testResults(net, dataloader, is_cuda, criterion):

    correct = 0
    total = 0
    running_loss = 0
    counter = 0
    net.is_training(False)

    with torch.no_grad():
        for data in dataloader:
            counter += 1
            images, labels = data

            if is_cuda:
                images = images.cuda() 
                labels = labels.cuda() 

            outputs = net(images)
            test_loss = criterion(outputs, labels)
            running_loss += test_loss.item()

            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.is_training(True)

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return (100 * correct / total), running_loss / counter


def train(name, net, epochs, trainloader, testloader, criterion, optimizer, is_cuda):
    print("im here")
    test_results = []
    train_loss = []


    result_percent, test_loss = testResults(net, testloader, is_cuda, criterion)
    best_result = result_percent
    test_results.append(test_loss)

    for epoch in range(epochs):
        print(epoch)
        running_loss = 0.0
        running_epoch_loss = 0.0
        i = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if is_cuda:
                inputs = inputs.cuda() 
                labels = labels.cuda() 

            # zero the parameter gradients
            optimizer.zero_grad()
            #net.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_epoch_loss += loss.item()
            if (i+1) % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                train_loss.append(running_loss / 10)
                running_loss = 0.0

        net.is_training(False)
        result_percent, test_loss = testResults(net, testloader, is_cuda, criterion)
        if result_percent > best_result:
            print("saving")
            best_result = result_percent
            torch.save(net.state_dict(), name + '.pt')
        print('[%d, %5d] Train epoch loss: %.3f' %
              (epoch + 1, i + 1, running_epoch_loss / i))
        print('[%d, %5d] Test epoch loss: %.3f' %
              (epoch + 1, i + 1, test_loss ))
        test_results.append(test_loss)


        running_epoch_loss = 0.0


    print('Finished Training')
    return train_loss, test_results


def train_net(name, net, optimizer, epoch, train_loader, test_loader, criterion):
    # creating the network
    print(net)

    # train
    return train(name, net, epoch, train_loader, test_loader, criterion, optimizer, is_cuda=False)


def print_graph(name, train_loss, test_results):

    test_res_multiplier = len(train_loss) / len(test_results)

    r = np.arange(start=0, stop=len(train_loss), step=1)
    r2 = np.arange(start=0, stop=len(test_results)*test_res_multiplier, step=test_res_multiplier)

    plt.plot(r, train_loss, color="blue")
    plt.plot(r2, test_results, color="green")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    # plt.show()
    plt.savefig(name + '.png')


def optimizer_builder(name, params, learning_rate):
    if name == 'SGD':
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    if name == 'Adam':
        return torch.optim.Adam(params, 1.0, weight_decay=1e-2)
    if name == 'RMSprop':
        return torch.optim.RMSprop(params, 0.1, weight_decay=1e-2)


def init_separated_folders():
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))]
                                         )
    image_datasets_train = datasets.ImageFolder(root='/Users/igorgumush/ml/new_data_50/train_data_32x32_10-20', transform=data_transforms)
    image_datasets_test = datasets.ImageFolder(root='/Users/igorgumush/ml/new_data_50/test_data_32x32_10-20', transform=data_transforms)

    print(image_datasets_train)
    print(image_datasets_test)

    print(image_datasets_train.classes)

    print(len(image_datasets_train.classes))

    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=32, shuffle=True)

    return trainloader, testloader, image_datasets_train

def init_single_folder():
    data_transforms = transforms.Compose([transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(root='/Users/igorgumush/Downloads/train_data_32x32/', transform=data_transforms)

    print(image_datasets)
    print(image_datasets.classes)

    valid_size = 0.2
    num_train = len(image_datasets)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_set = SubsetRandomSampler(train_idx)
    test_set = SubsetRandomSampler(valid_idx)

    print("Train set:", train_set)
    print("Test set: ", test_set)

    trainloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=32, sampler=train_set,
        num_workers=1, pin_memory=True, shuffle=False
    )

    testloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=32, sampler=test_set,
        num_workers=1, pin_memory=True, shuffle=False
    )

    return trainloader, testloader, image_datasets


def run():
    trainloader, testloader, image_datasets = init_separated_folders()

    size = len(image_datasets.classes)
    activation_func = torch.tanh
    # activation_func = F.relu

    net = CNN8(size, activation_func)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    criterion = nn.CrossEntropyLoss()

    name = "0001_08_CNN8_tanh_batch32_softmax_nodropout_v2"

    # net.load_state_dict(torch.load('001_08_CNN8_tanh_batch32_softmax_nodropout.pt'))

    train_loss, test_results = train_net(name, net, optimizer, 10000, trainloader, testloader, criterion)

    # name = "relu_200_08_CNN5"

    print_graph(name, train_loss, test_results)


def run_all():
    trainloader, testloader, image_datasets = init_single_folder()

    print("Num of cows: ", len(image_datasets.classes))
    size = len(image_datasets.classes)

    for learning_rate_name, learning_rate in [("001", 0.01), ("005", 0.05)]:
        activation_functions = [('relu', F.relu), ('sigmoid', torch.sigmoid), ('tanh', F.tanh)]
        for act_func_name, activation_func in activation_functions:
            optimizers = ['SGD', 'Adam', 'RMSprop']
            for optimizer_name in optimizers:
                # criterions = [ ('MSELoss', torch.nn.MSELoss()),
                # ('NLLLoss', torch.nn.NLLLoss()), ('MarginRankingLoss', torch.nn.MarginRankingLoss()), ,  ('HingeEmbeddingLoss', torch.nn.HingeEmbeddingLoss()
                criterions = [('cross entropy', nn.CrossEntropyLoss())]
                for criterion_name, criterion in criterions:
                    net = CNN(size, activation_func)
                    optimizer = optimizer_builder(optimizer_name, net.parameters(), learning_rate)

                    name = learning_rate_name + "_" + act_func_name + "_" + optimizer_name + "_" + criterion_name
                    print("-------------------------")
                    print(name)
                    print("-------------------------")

                    train_loss, test_results = train_net(net, optimizer, 50, trainloader, testloader, criterion)
                    print("***************")
                    print(train_loss)
                    print("***************")
                    print(test_results)

                    torch.save(net.state_dict(), name + '.pt')
                    print_graph(name, train_loss, test_results)


if __name__ == '__main__':
    run()