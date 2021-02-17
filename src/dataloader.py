from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms 

class MNISTDataLoader:
    """MNIST dataset"""

    train_loader = None
    test_loader = None
    train_dataset = None
    test_dataset = None 
    mytransform = None

    def __init__(self):

        self.mytransform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 mean = (0.1307,), std=(0.3081,))]
        )
        self.train_dataset = MNIST('../data/', train=True, download = True, transform=self.mytransform)
        self.test_dataset = MNIST('../data/', train=False, download = True, transform=self.mytransform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        

        
