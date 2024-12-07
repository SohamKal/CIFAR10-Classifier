import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets
import numpy as np
import random
import torchvision.models as models

# Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(100)

# TODO: Populate the dictionary with your hyperparameters for training


def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """

    config = {
        "batch_size": 128,
        "lr": 1e-4,
        "num_epochs": 20,
        "weight_decay": 1e-7,  # set to 0 if you do not want L2 regularization
        # Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)
        "save_criteria": 'accuracy',

    }

    return config


# TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO: Part 2 - Complete this with your Pretrained CNN architecture.
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)


# Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    test_transforms = None

    return train_dataset, valid_dataset, test_transforms

