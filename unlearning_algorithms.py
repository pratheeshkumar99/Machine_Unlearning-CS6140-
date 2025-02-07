import json
from models import TestModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import UnLearningData, process_dataloader, UnlearnerLoss
import numpy as np



class MUAlgorithm:
    """
    Base class for unlearning algorithms
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.device = device

        with open("config.json", "r") as f:
            self.config = json.load(f)

    def unlearn(self):
        pass

    def get_model(self):
        return self.model


class NaiveRetraining(MUAlgorithm):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_loader,
            retain_loader,
            forget_loader,
            device,
        )

    def unlearn(self):
        self.model.reset_model()
        print("\n\nPERFORMING NAIVE RETRAINING")
        print("RETRAINING MODEL")
        self.model.train(
            self.retain_loader,
            self.config["num_epochs_naive_retraining"],
            logging=False,
        )


class NegGrad(MUAlgorithm):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_loader,
            retain_loader,
            forget_loader,
            device,
        )

    # def unlearn(self):
    #     self.model.model.train()

    #     for epoch in range(self.config["num_epochs_negrad"]):
    #         forget_loader_iter = iter(self.forget_loader)
    #         for i, data in enumerate(self.retain_loader):
    #             try:
    #                 inputs_forget, labels_forget = next(forget_loader_iter)
    #             except StopIteration:
    #                 pass
    #             inputs_retain, labels_retain = data
    #             inputs_retain, labels_retain = inputs_retain.to(
    #                 self.device
    #             ), labels_retain.to(self.device)
    #             inputs_forget, labels_forget = inputs_forget.to(
    #                 self.device
    #             ), labels_forget.to(self.device)

    #             outputs_retain = self.model.model(inputs_retain)
    #             outputs_forget = self.model.model(inputs_forget)

    #             loss_retain = self.criterion(outputs_retain, labels_retain)
    #             loss_forget = -self.criterion(outputs_forget, labels_forget)

    #             loss = loss_retain + loss_forget
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    def unlearn(self):
        self.model.model.train()

        for epoch in range(self.config["num_epochs_negrad"]):
            forget_loader_iter = iter(self.forget_loader)
            for i, data in enumerate(self.retain_loader):
                try:
                    inputs_forget, labels_forget = next(forget_loader_iter)
                except StopIteration:
                    pass
                inputs_retain, labels_retain = data
                inputs_retain, labels_retain = inputs_retain.to(
                    self.device
                ), labels_retain.to(self.device)
                inputs_forget, labels_forget = inputs_forget.to(
                    self.device
                ), labels_forget.to(self.device)

                outputs_retain = self.model.model(inputs_retain)
                outputs_forget = self.model.model(inputs_forget)

                loss_retain = self.criterion(outputs_retain, labels_retain)
                loss_forget = -self.criterion(outputs_forget, labels_forget)

                loss = loss_retain + loss_forget
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for i, data in enumerate(self.retain_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class UNSIR(MUAlgorithm):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_loader,
            retain_loader,
            forget_loader,
            device,
        )

    def unlearn(self):
        pass


class SCRUB(MUAlgorithm):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_loader,
            retain_loader,
            forget_loader,
            device,
        )

    def unlearn(self):
        class KLDivLoss(nn.Module):
            def __init__(self, T):
                super(KLDivLoss, self).__init__()
                self.T = T

            def forward(self, y_s, y_t):
                p_s = F.log_softmax(y_s / self.T, dim=1)
                p_t = F.softmax(y_t / self.T, dim=1)
                loss = (
                    F.kl_div(p_s, p_t, reduction="batchmean")
                    * (self.T**2)
                    / y_s.shape[0]
                )
                return loss

        self.model.model.eval()
        student_model = TestModel(self.device)
        student_model.model.load_state_dict(self.model.model.state_dict())
        kldloss = KLDivLoss(4.0)

        for epoch in range(self.config["num_epochs_unsir"]):
            for i, data in enumerate(self.retain_loader):
                inputs_retain, labels_retain = data
                inputs_retain, labels_retain = inputs_retain.to(
                    self.device
                ), labels_retain.to(self.device)

                outputs_retain_student = student_model.model(inputs_retain)

                with torch.no_grad():
                    outputs_retain_teacher = self.model.model(inputs_retain)

                loss_1 = self.criterion(outputs_retain_student, labels_retain)
                loss_2 = kldloss(outputs_retain_student, outputs_retain_teacher)
                loss = loss_1 + loss_2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for i, data in enumerate(self.forget_loader):
                inputs_forget, labels_forget = data
                inputs_forget, labels_forget = inputs_forget.to(
                    self.device
                ), labels_forget.to(self.device)

                outputs_forget_student = student_model.model(inputs_forget)

                with torch.no_grad():
                    outputs_forget_teacher = self.model.model(inputs_forget)

                loss = -kldloss(outputs_forget_student, outputs_forget_teacher)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class BadTeachingUnlearning(MUAlgorithm):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        retain_loader,
        forget_loader,
        device,
        unlearning_teacher,
        full_trained_teacher,
        lr=0.0001,
        batch_size=64,
        KL_temperature=1,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_loader,
            retain_loader,
            forget_loader,
            device,
        )
        self.unlearning_teacher = unlearning_teacher
        self.full_trained_teacher = full_trained_teacher
        self.epochs = self.config["num_epochs_BadTeachingUnlearning"]
        self.lr = lr
        self.batch_size = batch_size
        self.KL_temperature = KL_temperature

    def unlearning_step(self, unlearn_data_loader):
        losses = []
        for batch in unlearn_data_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                full_teacher_logits = self.full_trained_teacher(x)
                unlearn_teacher_logits = self.unlearning_teacher(x)
            output = self.model(x)
            self.optimizer.zero_grad()
            loss = UnlearnerLoss(
                output=output,
                labels=y,
                full_teacher_logits=full_teacher_logits,
                unlearn_teacher_logits=unlearn_teacher_logits,
                KL_temperature=self.KL_temperature,
            )
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        return np.mean(losses)

    def unlearn(self):
        retain_data = process_dataloader(self.retain_loader, is_retain=True)
        forget_data = process_dataloader(self.forget_loader, is_retain=False)
        unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
        unlearning_loader = DataLoader(
            unlearning_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        
        self.unlearning_teacher.eval()
        self.full_trained_teacher.eval()

        for epoch in range(self.epochs):
            loss = self.unlearning_step(unlearning_loader)
            print(f"Epoch {epoch+1} Unlearning Loss: {loss}")
