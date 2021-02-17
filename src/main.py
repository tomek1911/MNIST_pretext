import dataloader
import pretext_models.model
import matplotlib.pyplot as plt

def main():

    loader = dataloader.MNISTDataLoader()
    examples = enumerate(loader.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()    
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()