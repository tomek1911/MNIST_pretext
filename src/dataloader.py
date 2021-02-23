import torch
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms 


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
        

        
