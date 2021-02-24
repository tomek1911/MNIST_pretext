import torch
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms 
import numpy as np
import copy


class MNISTDataLoader:
    """MNIST dataset"""

    train_loader = None
    test_loader = None
    validation_loader= None
    
    train_dataset = None
    validaton_dataset = None
    test_dataset = None 
    mytransform = None

    def __init__(self, path, isDownload, training_batch, test_batch):

        self.mytransform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 mean = (0.5,), std=(1,))]
        ) 
        train = MNIST(path, train=True, download = isDownload, transform=self.mytransform)
        self.train_dataset, self.validaton_dataset = torch.utils.data.random_split(
        train, [50000, 10000], generator=torch.Generator().manual_seed(1))
        
        self.test_dataset = MNIST(path, train=False, download = isDownload, transform=self.mytransform)
       
        self.train_loader = DataLoader(self.train_dataset, batch_size=training_batch, shuffle=True)
        self.validation_loader = DataLoader(self.validaton_dataset, batch_size=test_batch, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch, shuffle=False)

class CIFAR10DataLoader:
    """CIFAR10 dataset"""

    train_loader = None
    test_loader = None
    validation_loader= None
    
    train_dataset = None
    validaton_dataset = None
    test_dataset = None 
    mytransform = None

    def __init__(self, path, isDownload, training_batch, test_batch):

        self.normalization = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        self.denormalization = ((-0.4914, -0.4822, -0.4465), (1/0.247, 1/0.243, 1/0.261))

        self.mytransform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 mean = (0.5,0.5,0.5), std=(0.5,0.5,0.5))]
        ) 

        train = CIFAR10(path, train=True, download = isDownload, transform=self.mytransform)
        self.train_dataset, self.validaton_dataset = torch.utils.data.random_split(
        train, [42000, 8000], generator=torch.Generator().manual_seed(1))
        
        self.test_dataset = CIFAR10(path, train=False, download = isDownload, transform=self.mytransform)
       
        self.train_loader = DataLoader(self.train_dataset, batch_size=training_batch, shuffle=True)
        self.validation_loader = DataLoader(self.validaton_dataset, batch_size=test_batch, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch, shuffle=False)

        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def unnorm_recreate(self, im_as_var):
        reverse_mean = [-0.5, -0.5, -0.5]
        reverse_std = [1/0.5, 1/0.5, 1/0.5]
        recreated_im = copy.copy(im_as_var.data.numpy())
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        return recreated_im
            

        
