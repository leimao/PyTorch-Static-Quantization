# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://pytorch.org/docs/stable/quantization.html
# https://pytorch.org/docs/master/torch.quantization.html#torch.quantization.prepare
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# https://leimao.github.io/blog/PyTorch-Distributed-Training/

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

import copy
import numpy as np

from resnet import resnet18

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

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

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device):

    # The training configurations were not carefully selected.
    learning_rate = 1e-2
    num_epochs = 20

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(test_loader.dataset)
        eval_accuracy = running_corrects / len(test_loader.dataset)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    
    return model

def calibrate_model(model, loader, device):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model



def create_model():

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = resnet18(pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    return model

class ModifiedResNet18(torch.nn.Module):
    def __init__(self, model_fp32):
        super(ModifiedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
'''
def is_equivalent(model_1, model_2, num_tests=100):

    for _ in range(num_tests):
        x = torch.rand(size=(1,3,32,32))
        y1 = model_1(x)
        y2 = model_2(x)
        if torch.all(torch.eq(y1, y2)) == False:
            print(y1)
            print(y2)
            return False

    return True
'''

def is_equivalent(model_1, model_2, device, num_tests=100):

    # model_1.to(device)
    # model_2.to(device)

    for i in range(num_tests):
        x = torch.rand(size=(1,3,32,32)).to(device)
        y1 = model_1(x)
        y2 = model_2(x)
        if torch.all(torch.eq(y1, y2)) == False:
            print("Test sample: {}".format(i))
            print(y1)
            print(y2)
            return False

    return True

if __name__ == "__main__":

    random_seed=0
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = "resnet18_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=0)

    # Create an untrained model.
    model = create_model()

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)
    
    # Train model.
    # model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device)
    # Save model.
    # save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)
    # Currently, static quantization does not support CUDA device
    #fused_model.to(cpu_device)

    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model, [['conv1', 'bn1', 'relu']], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
                torch.quantization.fuse_modules(basic_block, [['conv1', 'bn1', 'relu1']], inplace=True)
    # for module_name, module in fused_model.named_children():
    #     print("=" * 50)
    #     print(module_name)
    #     print(module)
    #     print("=" * 50)



    model.eval()
    fused_model.eval()

    # Print fused model
    print(model)
    print("="* 100)
    print(fused_model)


    # Model and fused model should be equivalent
    assert is_equivalent(model_1=model, model_2=fused_model, device=cpu_device), "Fused model is not equivalent to the fused model!"

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    # modified_model = ModifiedResNet18(model_fp32=fused_model)
    # modified_model.to(cpu_device)
    # modified_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    # calibration_prepared_model = torch.quantization.prepare(modified_model)

    # Use training data for calibration
    # calibrate_model(model=calibration_prepared_model, loader=train_loader, device=cpu_device)

    # quantized_model = torch.quantization.convert(calibration_prepared_model)

    # assert is_equivalent(model_1=model, model_2=quantized_model), "Fused model is not equivalent to the fused model!"







    #modified_model = ModifiedResNet18(model_fp32=model)




    #print(modified_model)


    #model_fp32_fused = torch.quantization.fuse_modules(modified_model, [['conv1', 'bn1', 'relu']])

    #print(model_fp32_fused)

    # for m in modified_model.modules():
    #     if type(m) == "layer1":
    #         print(m)

    # for name, m in modified_model.named_modules():
    #     print("-------------------")
    #     print(name)
    #     print(m)
    #     print("-------------------")

    # # Fuse the model in place rather manually.
    # for module_name, module in modified_model.named_children():
    #     if module_name == "model_fp32":
            
    #         for submodule_name, submodule in module.named_children():
    #             if "layer" in submodule_name:
    #                 # print("-------------------")
    #                 # print(submodule_name)
    #                 # print(submodule)
    #                 # print("-------------------")

    #                 for basic_block_name, basic_block in submodule.named_children():
    #                     # print("-------------------")
    #                     # print(basic_block_name)
    #                     # print(basic_block)
    #                     # print("-------------------")
    #                     torch.quantization.fuse_modules(basic_block, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)

    # # Examine the modified model
    # print(modified_model)

    # for module_name, module in modified_model.named_children():
    #     if module_name == "model_fp32":
    #         for submodule_name, submodule in module.named_children():
    #             if "layer" in submodule_name:
    #                 # print("-------------------")
    #                 # print(submodule_name)
    #                 # print(submodule)
    #                 # print("-------------------")

    #                 for basic_block_name, basic_block in submodule.named_children():
    #                     print("-------------------")
    #                     print(basic_block_name)
    #                     print(basic_block)
    #                     print("-------------------")

    #print(model)

