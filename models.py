import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class TestModel:
    def __init__(self, device):
        with open("config.json", "r") as f:
            self.config = json.load(f)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.learning_rate = self.config["init_learning_rate"]
        with open("config.json", "r") as f:
            self.config = json.load(f)
        self.__setup()

    def __setup(self):
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 8)
        model = model.to(self.config["device"])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def get_model(self):
        return self.model

    def adjust_learning_rate(self, epoch):
        lr = self.learning_rate
        if epoch >= 10:
            lr /= 10
        if epoch >= 20:
            lr /= 10
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, train_loader, num_epochs, logging=True):
        self.model.train()

        for epoch in range(num_epochs):
            self.adjust_learning_rate(epoch)
            total_loss = 0.0
            total_correct = 0
            total = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.config["device"]), labels.to(
                    self.config["device"]
                )
                total += labels.shape[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                total_correct += torch.sum(
                    torch.argmax(outputs, dim=1) == labels
                ).item()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            if logging:
                print(
                    f"Epoch {epoch + 1}, Loss: {total_loss / total}, Accuracy: {total_correct / total}"
                )

    def evaluate(self, test_loader):
        self.model.eval()
        total_correct = 0
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.config["device"]), labels.to(
                    self.config["device"]
                )
                total += labels.shape[0]
                outputs = self.model(inputs)
                total_correct += torch.sum(
                    torch.argmax(outputs, dim=1) == labels
                ).item()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        print(f"Loss: {total_loss / total}, Accuracy: {total_correct / total}")

    def save_model(self, path="./saved_models/base_trained.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="./saved_models/base_trained.pt"):
        self.__setup()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def reset_model(self):
        self.__setup()
