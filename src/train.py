import time
import torch.optim
import torch.nn.functional as F

from dataloader import MNISTDataLoader
from pretext_models.pilot_model import Net
import matplotlib.pyplot as plt

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

loger_interval = 100
n_epochs = 5
training_batch = 32
save_state = False

use_cuda = torch.cuda.is_available()
my_device = torch.device("cuda:0" if use_cuda else "cpu")
print('Using: ', my_device)

network = Net().to(device=my_device)
optimizer = torch.optim.SGD(network.parameters(), lr = 0.01, momentum=0.5, weight_decay=1e-3)
loader = MNISTDataLoader('../data', False, training_batch)
train_loader = loader.train_loader
test_loader = loader.test_loader

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data, target = data.to(my_device), target.to(my_device)
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % loger_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*training_batch) + ((epoch-1)*len(train_loader.dataset)))
      if save_state:  
        torch.save(network.state_dict(), './results/model.pth')
        torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(my_device), target.to(my_device)  
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
start_time = time.time()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

print('Train time: {:.2f}s'.format(time.time() - start_time))

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

plt.show()