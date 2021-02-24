
import argparse
import time
import torch.nn
import torch.optim
import torch.nn.functional as F
from torchsummary import summary

from dataloader import CIFAR10DataLoader, MNISTDataLoader
from pretext_models.models import Net
from pretext_models.models import myVgg
import utils.plots as pls
from utils.lr_range_finder import LRFinder

torch.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--n_epochs", help="number of learning epochs", type=int, default=14
)
parser.add_argument(
    "-b", "--batch_size", help="size of minibatch", type=int, default=128
)
parser.add_argument(
    "--min_lr", help="base minimal value of lerning rate for CyclicLR", type=float, default=0.05
)
parser.add_argument(
    "--max_lr", help="base minimal value of lerning rate for CyclicLR", type=float, default=0.25
)
parser.add_argument(
    "-m", "--momentum", help="base minimal momentum value for CyclicLR", type=float, default=0.5
)
parser.add_argument(
    "-c", "--cycles", help="how many cycles of CyclicLR, default one cycle", type=int, default=1
)
parser.add_argument(
    "-d", "--dataset", help="chose dataset: MNIST, CIFAR10", type=str, default="MNIST"
)
parser.add_argument(
    "-l", "--lr_find", help="find best range for cyclic learning rate", type=bool, default=False
)
args = parser.parse_args()

go_deeper = False
save_state = True
loger_interval = 10

n_epochs = args.n_epochs
batch_size = args.batch_size
min_lr = args.min_lr
max_lr = args.max_lr
base_momentum = args.momentum
cycles = args.cycles
dataset = args.dataset
is_lr_find = args.lr_find

print(f"Training config:\n\t dataset: {dataset}, n_epochs: {n_epochs}, batch_size: {batch_size}, min_lr: {min_lr}, max_lr: {max_lr}, base_momentum: {base_momentum}.")

use_cuda = torch.cuda.is_available()
my_device = torch.device("cuda:0" if use_cuda else "cpu")
print('Using device: ', my_device)

network = myVgg().to(device=my_device)

loader = None
network = None
if dataset == "MNIST":
  network = Net().to(device=my_device)
  summary(network, (1, 28, 28))
  loader = MNISTDataLoader('./data', True, batch_size, batch_size)
elif dataset == "CIFAR10":
  if go_deeper:
    network = myVgg().to(device=my_device)
    summary(network, (3, 32, 32))
  else:
    network = Net(channels=3).to(device=my_device)
    summary(network, (3, 32, 32))
  loader = CIFAR10DataLoader('./data', True, batch_size, batch_size)

if (loader or network) is None:
  raise Exception("Error: Missing dataset loader or network model!")

train_loader = loader.train_loader
test_loader = loader.test_loader
validation_loader = loader.validation_loader
train_dataset_size = len(train_loader.dataset)
validation_dataset_size = len(validation_loader.dataset)
test_dataset_size = len(test_loader.dataset)

if is_lr_find:
  optimizer_lr = torch.optim.SGD(network.parameters(), lr=1e-4)
  lr_finder = LRFinder(network, optimizer_lr, torch.nn.CrossEntropyLoss(), my_device)
  lr_finder.range_test(train_loader, end_lr=1, num_iter=50)
  lr_finder.plot()
  # network = Net(channels=3).to(device=my_device)

optimizer = torch.optim.SGD(network.parameters(), lr=min_lr, momentum= 0.5)
step_up = (train_dataset_size / (batch_size))*(n_epochs/2)/cycles
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, cycle_momentum = True, base_momentum=0.5, max_momentum=0.9, step_size_up = int(step_up))

if dataset == "MNIST":
    pls.plot_dataset_sample(loader, filename="mnist_samples", grayscale=True)
elif dataset == "CIFAR10":
    pls.plot_dataset_sample(loader, filename="cifar_samples", grayscale=False)


train_losses = []
valid_losses = []
test_losses = []

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data, target = data.to(my_device), target.to(my_device)
    output = network(data)
    loss = F.cross_entropy(output, target) # negative log-likelihood loss + log_softmax
    loss.backward()
    optimizer.step()
    scheduler.step()

    if batch_idx % loger_interval == 0:
      optimizer
      print(f'Train epoch: {epoch} seen [{batch_idx * train_loader.batch_size}/{train_dataset_size}]\tLoss: {loss.item():.3f}')
      train_losses.append(loss.item())
      
      if save_state:  
        torch.save(network.state_dict(), './results/model.pth')
        torch.save(optimizer.state_dict(), './results/optimizer.pth')

def validation():
  valid_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in validation_loader:
      data, target = data.to(my_device), target.to(my_device)  
      output = network(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
      valid_loss += F.cross_entropy(output, target, reduction='sum').item()
    valid_loss /= validation_dataset_size
    valid_losses.append(valid_loss)
    print(f'\nValidation set: Avg. loss: {valid_loss:.3f}, Accuracy: {correct}/{validation_dataset_size} ({100. * correct/validation_dataset_size:.1f}%)\n')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(my_device), target.to(my_device)  
      output = network(data)
      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print(f'Test set: Avg. loss: {test_loss:.3f}, Accuracy: {correct}/{test_dataset_size} ({100. * correct/test_dataset_size:.1f}%)\n')

start_time = time.time()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  validation()

test()
print('Train time: {:.2f}s'.format(time.time() - start_time))

pls.plot_training_results(train_losses, valid_losses, n_epochs,"cifar_training")

## check results for classes

if dataset == "CIFAR10":

  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(my_device), target.to(my_device) 
          outputs = network(data)
          _, predicted = torch.max(outputs, 1)
          c = (predicted == target).squeeze()
          for i in range(4):
              label = target[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1


  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (
          loader.classes[i], 100 * class_correct[i] / class_total[i]))