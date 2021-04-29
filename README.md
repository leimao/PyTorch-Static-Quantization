# PyTorch Static Quantization

## Introduction

PyTorch post-training static quantization example for ResNet.

## Usages

### Build Docker Image

```
$ docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:1.8.1 .
```

### Run Docker Container

```
$ docker run -it --rm --gpus device=0 --ipc=host -v $(pwd):/mnt pytorch:1.8.1
```

### Run ResNet

```
$ python cifar.py
```

## References

* [PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
* [PyTorch CIFAR10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
* [PyTorch Distributed Training](https://leimao.github.io/blog/PyTorch-Distributed-Training/)
