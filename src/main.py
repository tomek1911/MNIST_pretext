import torch
import torch.optim as optim

import dataloader
import matplotlib.pyplot as plt

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

network = None
optimizer = None
loader = None

def main():
    # examples = enumerate(loader.test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # fig = plt.figure()    
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    pass

if __name__ == "__main__":
    main()