import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(train_losses, valid_losses, n_epochs):

    plt.figure()
    training_x = list(range(1, len(train_losses) + 1))
    plt.plot(training_x, train_losses, color='blue')
    validation_x = np.arange(len(train_losses)/n_epochs, len(train_losses)+1, len(train_losses)/n_epochs).tolist()
    plt.plot(validation_x, valid_losses, color='red')
    plt.legend(['Train Loss', 'Valid Loss'], loc='upper right')
    plt.xlabel('training samples')
    plt.ylabel('negative log likelihood loss')
    plt.savefig("results/plot.png")


def plot_dataset_sample(data_loader):
    examples = enumerate(data_loader)
    _, (example_data, example_targets) = next(examples)
    
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("results/data_samples.png")