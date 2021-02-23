import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

class DataLoaderIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):

        inputs, labels, *_ = batch_data
        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self.inputs_labels_from_batch(batch)

class TrainDataLoaderIter(DataLoaderIter):
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def __next__(self):
        try:
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)
        return inputs, labels

class LRFinder(object):
    def __init__(
        self,
        model,      
        optmizer,
        criterion,
        device = None,
    ):
        
        self.optimizer = optmizer
        self.model = model
        self.criterion = criterion
        self.model_device = model

        if device:
            self.device = next(self.model.parameters()).device
        else:
            self.device = self.model_device
    
    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr=None,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
        accumulation_steps=1
    ):

        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model.to(self.device)

        if start_lr:
            self._set_learning_rate(start_lr)
        
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)

        train_iter = TrainDataLoaderIter(train_loader)

        for iteration in tqdm(range(num_iter)):
                # Train on batch and retrieve loss
                loss = self._train_batch(
                    train_iter,
                    accumulation_steps,
                )
       
                # Update the learning rate
                self.history["lr"].append(lr_schedule.get_lr()[0])
                lr_schedule.step()

                # Track the best loss and smooth it if smooth_f is specified
                if iteration == 0:
                    self.best_loss = loss
                else:
                    if smooth_f > 0:
                        loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                    if loss < self.best_loss:
                        self.best_loss = loss

                # Check if the loss has diverged; if it has, stop the test
                self.history["loss"].append(loss)
                if loss > diverge_th * self.best_loss:
                    print("Stopping early, the loss has diverged")
                    break

        print("Learning rate search finished.")

    def _train_batch(self, train_iter, accumulation_steps):
        self.model.train()
        total_loss = None 
        self.optimizer.zero_grad()

        for i in range(accumulation_steps):
            inputs, labels = next(train_iter)
            inputs, labels = self._move_to_device(
                inputs, labels
            )

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)           
            loss /= accumulation_steps
            loss.backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        self.optimizer.step()

        return total_loss.item()
        
    def _move_to_device(self, inputs, labels):

        def move(obj, device):
            if hasattr(obj, "to"):
                return obj.to(device)
            elif isinstance(obj, tuple):
                return tuple(move(o, device) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device)
        labels = move(labels, self.device)
        return inputs, labels
        
    def plot(
        self,
        log_lr=True,
        show_lr=None,
        ax=None,
        suggest_lr=True,
    ):

        lrs = self.history["lr"]
        losses = self.history["loss"]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        # Plot the suggested LR
        if suggest_lr:
            # 'steepest': the point with steepest gradient (minimal gradient)
            print("LR suggestion: steepest gradient")
            min_grad_idx = None
            try:
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
            except ValueError:
                print(
                    "Failed to compute the gradients, there might not be enough points."
                )
            if min_grad_idx is not None:
                print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
                ax.scatter(
                    lrs[min_grad_idx],
                    losses[min_grad_idx],
                    s=50,
                    marker="x",
                    color="red",
                    zorder=3,
                    label="steepest gradient",
                )
                ax.legend()

        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        if fig is not None:
            plt.savefig("results/lr_find.png")

        if suggest_lr and min_grad_idx is not None:
            return ax, lrs[min_grad_idx]
        else:
            return ax

class LinearLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):


    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
            




