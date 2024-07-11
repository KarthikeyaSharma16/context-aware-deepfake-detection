import torch
import torch.nn as nn
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import watch, log_model
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os

class Trainer:
    def __init__(
        self,
        model,
        trainset,
        testset,
        num_epochs=5,
        batch_size=16,
        init_lr=1e-3,
        device="cpu",
        checkpoint_dir="ckpts",
        experiment=None
    ):
        self.model = model.to(device)
        self.trainset = trainset
        self.testset = testset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment = experiment

        self.train_loss_per_iteration = []
        self.train_accuracy_per_iteration = []
        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.test_loss_per_epoch = []
        self.test_accuracy_per_epoch = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        loss_fn = nn.BCELoss()
        optimizer = Adamax(self.model.parameters(), lr=self.init_lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.experiment.log_current_epoch(epoch)
            self.model.train()
            running_loss = 0
            correct = 0
            total = 0
            with tqdm(trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
                for idx, data in enumerate(tepoch):
                    # print('Epoch: {epoch}, Batch/Iter: {idx}')
                    # print(f'Dictionary Data: {data}')
                    id, mask, video, labels = data['id'], data['mask'], data['video'], data['label']
                    id, mask, video, labels = id.to(self.device), mask.to(self.device), video.to(self.device), labels.to(self.device)
                    # print(f'In Train, video shape = {video.shape}')
                    optimizer.zero_grad()
                    outputs = self.model(id, mask, video)

                    loss = loss_fn(outputs, labels.unsqueeze(1).float())
                    loss.backward()
                    optimizer.step()

                    total += len(labels)
                    correct += ((outputs > 0.5).float() == labels.unsqueeze(1)).sum().item()
                    running_loss += loss.item()
                    self.train_loss_per_iteration.append(loss.item())
                    self.train_accuracy_per_iteration.append(correct / total)
                    self.experiment.log_metric("train_loss_per_iteration", loss.item())
                    self.experiment.log_metric("train_accuracy_per_iteration", correct/total)
                    tepoch.set_postfix(
                        loss=running_loss / (idx + 1), accuracy=correct / total
                    )
            scheduler.step()
            self.train_loss_per_epoch.append(running_loss / len(trainloader))
            self.train_accuracy_per_epoch.append(correct / total)
            self.experiment.log_metric("train_loss_per_epoch", running_loss/len(trainloader))
            self.experiment.log_metric("train_accuracy_per_epoch", correct/total)

            # validation
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                test_correct = 0
                test_total = 0
                for idx, data in enumerate(testloader):
                    id, mask, video, labels = data['id'], data['mask'], data['video'], data['label']
                    id, mask, video, labels = id.to(self.device), mask.to(self.device), video.to(self.device), labels.to(self.device)
                    outputs = self.model(id, mask, video)
                    loss = loss_fn(outputs, labels.unsqueeze(1).float())
                    test_loss += loss.item()
                    test_total += len(labels)
                    test_correct += ((outputs > 0.5).float() == labels.unsqueeze(1)).sum().item()
                print(
                    f"Epoch {epoch + 1}: Validation Loss: {test_loss / len(testloader):.4f}, Validation Accuracy: {test_correct / test_total:.4f}"
                )
                self.test_loss_per_epoch.append(test_loss / len(testloader))
                self.test_accuracy_per_epoch.append(test_correct / test_total)
                self.experiment.log_metric("Validation loss per epoch", test_loss/len(testloader))
                self.experiment.log_metric("Validation accuracy per epoch", test_correct/test_total)

            # save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss_per_iteration": self.train_loss_per_iteration,
                "train_accuracy_per_iteration": self.train_accuracy_per_iteration,
                "train_loss_per_epoch": self.train_loss_per_epoch,
                "train_accuracy_per_epoch": self.train_accuracy_per_epoch,
                "test_loss_per_epoch": self.test_loss_per_epoch,
                "test_accuracy_per_epoch": self.test_accuracy_per_epoch,
            }
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pt"))
            log_model(self.experiment, checkpoint, model_name="hmcan")

            # save best weights
            if test_loss / len(testloader) < best_val_loss:
                best_val_loss = test_loss / len(testloader)
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_weights.pt"))

    def get_training_history(self):
        return (
            self.train_loss_per_iteration,
            self.train_accuracy_per_iteration,
            self.train_loss_per_epoch,
            self.train_accuracy_per_epoch,
            self.test_loss_per_epoch,
            self.test_accuracy_per_epoch,
        )

    def predict(self, testloader):
        self.model.eval()
        predict_probs = []
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for data in testloader:
                id, mask, video, labels = data['id'], data['mask'], data['video'], data['label']
                id, mask, video, labels = id.to(self.device), mask.to(self.device), video.to(self.device), labels.to(self.device)
                
                outputs = self.model(id, mask, video)
                predict_probs.append(outputs)
                predictions.append((outputs > 0.5).float())
                ground_truth.append(labels)

        return (
            torch.cat(predict_probs).cpu(),
            torch.cat(predictions).cpu(),
            torch.cat(ground_truth).cpu(),
        )