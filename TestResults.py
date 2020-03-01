
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from CNN3 import CNN3
from CNN import CNN
from CNN5 import CNN5
from CNN6 import CNN6
from CNN8 import CNN8

def testResults(net, dataloader, is_cuda, criterion, size):
    correct = 0
    total = 0
    running_loss = 0
    counter = 0

    histogram = {}
    wrong_histogram = {}
    mistakes = []
    net.is_training(False)

    for i in range(size):
        wrong_histogram[i] = 0
        histogram[i] = 0


    with torch.no_grad():
        for data in dataloader:
            counter = counter + 1
            images, labels = data

            if is_cuda:
                images = images.cuda()  # -- for GPU
                labels = labels.cuda()  # -- for GPU

            outputs = net(images)
            test_loss = criterion(outputs, labels)
            running_loss += test_loss.item()

            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted)
            correct += (predicted == labels).sum().item()

            if predicted == labels:
                histogram[int(labels)] = histogram[int(labels)] + 1
            else:
                wrong_histogram[int(labels)] = wrong_histogram[int(labels)] + 1
                mistakes.append( (int(labels), int(predicted)) )

    for i in range(size):
        print(str(i) + ": " + str(histogram[i] / (histogram[i] + wrong_histogram[i])))

    net.is_training(True)

    print("correct:", correct, " out of ", len(dataloader.dataset))
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return (100 * correct / total), running_loss / counter, mistakes


def run():
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))]
                                         )
    image_datasets_test = datasets.ImageFolder(root='/Users/igorgumush/ml/new_data_50/test_data_32x32_10-20', transform=data_transforms)
    testloader = torch.utils.data.DataLoader(image_datasets_test, batch_size=1, shuffle=False)

    size = len(image_datasets_test.classes)
    activation_func = torch.tanh #F.relu #

    # net = CNN6(size, activation_func)
    # net.load_state_dict(torch.load('storev2/tanh_200_08_CNN6_softmax_v5.pt'))
    net = CNN8(size, activation_func)
    net.load_state_dict(torch.load('0001_08_CNN8_tanh_batch32_softmax.pt'))

    criterion = nn.CrossEntropyLoss()
    _, _, mistakes = testResults(net, testloader, False, criterion, size)

    print_mistakes_histogram(mistakes, size)


def print_mistakes_histogram(mistakes, num_classes, threshold=20):
    print(mistakes)
    x = []
    for class_idx in range(num_classes):
        for mistake in mistakes:
            if mistake[0] == class_idx:
                x.append(mistake[1])

        if len(x) > threshold:
            plt.hist(x, bins=num_classes)
            plt.show()

if __name__ == '__main__':
    run()