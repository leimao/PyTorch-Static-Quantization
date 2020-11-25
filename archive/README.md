# PyTorch Static Quantization

```
docker build -f docker/pytorch.Dockerfile --no-cache --tag=pytorch:1.7.0 .
```

```
docker run -it --rm --gpus device=0 -v $(pwd):/mnt pytorch:1.7.0
```

```
python cifar.py
```

```
Epoch: 00 Train Loss: 1.683 Train Acc: 0.390 Eval Loss: 1.482 Eval Acc: 0.476
Epoch: 01 Train Loss: 1.332 Train Acc: 0.521 Eval Loss: 1.243 Eval Acc: 0.565
Epoch: 02 Train Loss: 1.167 Train Acc: 0.584 Eval Loss: 1.091 Eval Acc: 0.618
Epoch: 03 Train Loss: 1.037 Train Acc: 0.632 Eval Loss: 1.041 Eval Acc: 0.654
Epoch: 04 Train Loss: 0.967 Train Acc: 0.661 Eval Loss: 0.933 Eval Acc: 0.676
Epoch: 05 Train Loss: 0.906 Train Acc: 0.681 Eval Loss: 0.856 Eval Acc: 0.702
Epoch: 06 Train Loss: 0.852 Train Acc: 0.697 Eval Loss: 0.881 Eval Acc: 0.705
Epoch: 07 Train Loss: 0.809 Train Acc: 0.714 Eval Loss: 0.797 Eval Acc: 0.725
Epoch: 08 Train Loss: 0.773 Train Acc: 0.729 Eval Loss: 0.749 Eval Acc: 0.738
Epoch: 09 Train Loss: 0.745 Train Acc: 0.737 Eval Loss: 0.828 Eval Acc: 0.721
Epoch: 10 Train Loss: 0.715 Train Acc: 0.750 Eval Loss: 0.748 Eval Acc: 0.739
Epoch: 11 Train Loss: 0.694 Train Acc: 0.758 Eval Loss: 0.696 Eval Acc: 0.757
Epoch: 12 Train Loss: 0.667 Train Acc: 0.767 Eval Loss: 0.727 Eval Acc: 0.752
Epoch: 13 Train Loss: 0.638 Train Acc: 0.775 Eval Loss: 0.682 Eval Acc: 0.764
Epoch: 14 Train Loss: 0.622 Train Acc: 0.781 Eval Loss: 0.688 Eval Acc: 0.766
Epoch: 15 Train Loss: 0.611 Train Acc: 0.786 Eval Loss: 0.699 Eval Acc: 0.759
Epoch: 16 Train Loss: 0.589 Train Acc: 0.795 Eval Loss: 0.703 Eval Acc: 0.758
Epoch: 17 Train Loss: 0.577 Train Acc: 0.797 Eval Loss: 0.654 Eval Acc: 0.776
Epoch: 18 Train Loss: 0.553 Train Acc: 0.807 Eval Loss: 0.672 Eval Acc: 0.773
Epoch: 19 Train Loss: 0.542 Train Acc: 0.808 Eval Loss: 0.621 Eval Acc: 0.793
Epoch: 20 Train Loss: 0.525 Train Acc: 0.814 Eval Loss: 0.618 Eval Acc: 0.790
Epoch: 21 Train Loss: 0.510 Train Acc: 0.820 Eval Loss: 0.639 Eval Acc: 0.784
Epoch: 22 Train Loss: 0.499 Train Acc: 0.824 Eval Loss: 0.605 Eval Acc: 0.793
Epoch: 23 Train Loss: 0.492 Train Acc: 0.828 Eval Loss: 0.591 Eval Acc: 0.798
Epoch: 24 Train Loss: 0.473 Train Acc: 0.835 Eval Loss: 0.613 Eval Acc: 0.794
Epoch: 25 Train Loss: 0.465 Train Acc: 0.838 Eval Loss: 0.632 Eval Acc: 0.790
Epoch: 26 Train Loss: 0.454 Train Acc: 0.839 Eval Loss: 0.626 Eval Acc: 0.795
Epoch: 27 Train Loss: 0.442 Train Acc: 0.842 Eval Loss: 0.584 Eval Acc: 0.809
Epoch: 28 Train Loss: 0.430 Train Acc: 0.846 Eval Loss: 0.625 Eval Acc: 0.797
Epoch: 29 Train Loss: 0.425 Train Acc: 0.850 Eval Loss: 0.591 Eval Acc: 0.808
```