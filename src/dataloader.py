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

    def __init__(self, path, isDownload, training_batch):

        self.mytransform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(
                 mean = (0.1307,), std=(0.3081,))]
        )
        self.train_dataset = MNIST(path, train=True, download = isDownload, transform=self.mytransform)
        self.test_dataset = MNIST(path, train=False, download = isDownload, transform=self.mytransform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=training_batch, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1024, shuffle=False)
        

        
