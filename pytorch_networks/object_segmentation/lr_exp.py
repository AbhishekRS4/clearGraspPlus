import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from train import PolynomialLR
from modeling import deeplab

def main():
    learning_rate = 1e-2
    learning_rate = 9.81e-4
    weight_decay = 1e-4
    num_epochs = 12
    power = 0.25
    seg_model = deeplab.DeepLab(num_classes=2, backbone="drn_psa", sync_bn=True,
                            freeze_bn=False)  # output stride is 8 for drn_psa
    optimizer = torch.optim.SGD(
        seg_model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    lr_scheduler = PolynomialLR(
        optimizer, num_epochs+1, power=power,
    )
    for epoch in range(1, num_epochs+1):
        lr_epoch = lr_scheduler.get_last_lr()
        print(f"epoch: {epoch}, learning_rate: {lr_epoch[0]:.7f}")
        lr_scheduler.step()
    return

if __name__ == "__main__":
    main()
