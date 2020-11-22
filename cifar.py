# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://pytorch.org/docs/stable/quantization.html
# https://pytorch.org/docs/master/torch.quantization.html#torch.quantization.prepare
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# https://leimao.github.io/blog/PyTorch-Distributed-Training/

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

import numpy as np

def train_cifar_classifier(device):

    # The training configurations were not carefully selected.
    random_seed = 0
    num_workers = 8
    train_batch_size = 128
    eval_batch_size = 256
    learning_rate = 1e-2
    num_epochs = 30

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform) 
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()

    model = torchvision.models.resnet18(pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_set)
        train_accuracy = running_corrects / len(train_set)

        # Evaluation
        model.eval()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(test_set)
        eval_accuracy = running_corrects / len(test_set)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    
    return model






if __name__ == "__main__":

    device = torch.device("cuda:0")
    
    model = train_cifar_classifier(device=device)