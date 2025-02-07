import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset, Subset
from sklearn.mixture import GaussianMixture
import numpy as np
from models import TestModel
from mu_datasets import get_dataset
import numpy as np
from sklearn import linear_model, model_selection


class UnlearningMetric:
    """
    Base class for unlearning metrics
    """

    def __init__(
        self,
        base_model,
        unlearnt_model,
        train_data,
        forget_data,
        retain_data,
        test_data,
    ):
        self.base_model = base_model
        self.unlearnt_model = unlearnt_model
        self.train_data = train_data
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.test_data = test_data
        with open("config.json", "r") as f:
            self.config = json.load(f)

    def evaluate(self):
        pass


class LossBasedMIA(UnlearningMetric):
    def __init__(self):
        super().__init__()

    def evaluate(base_model, unlearnt_model, train_data, forget_data, retain_data):
        pass


class CaliberatedLossBasedClassifierMIA(UnlearningMetric):
    def __init__(
        self,
        base_model,
        unlearnt_model,
        train_data,
        forget_data,
        retain_data,
        test_data,
    ):
        super().__init__(
            base_model, unlearnt_model, train_data, forget_data, retain_data, test_data
        )

    def compute_loss(self, model, dataset):
        model.model.eval()
        losses = np.array([])
        # criterion = nn.CrossEntropyLoss(reduction="none")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, targets = data
                inputs, targets = inputs.to(self.config["device"]), targets.to(
                    self.config["device"]
                )
                outputs = model.model(inputs)
                batch_losses = model.criterion(outputs, targets)

                losses = np.append(losses, [batch_losses.cpu().numpy()])
        return np.array(losses)

    def simple_mia(self, sample_loss, members, n_splits=10, random_state=0):
        unique_members = np.unique(members)
        if not np.all(unique_members == np.array([0, 1])):
            raise ValueError("members should only have 0 and 1s")

        attack_model = linear_model.LogisticRegression()
        cv = model_selection.StratifiedShuffleSplit(
            n_splits=n_splits, random_state=random_state
        )
        return model_selection.cross_val_score(
            attack_model, sample_loss, members, cv=cv, scoring="accuracy"
        )

    def evaluate(self):

        forget_losses = self.compute_loss(
            self.unlearnt_model,
            self.forget_data,
        )
        unseen_losses = self.compute_loss(self.unlearnt_model, self.test_data)

        np.random.shuffle(forget_losses)
        forget_losses = forget_losses[: len(unseen_losses)]

        samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
        labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

        mia_scores = self.simple_mia(samples_mia, labels_mia)
        forgetting_score = abs(0.5 - mia_scores.mean())

        return {"MIA": mia_scores.mean(), "Forgeting Score": forgetting_score}


class LiRAMIA(UnlearningMetric):
    def __init__(
        self,
        base_model,
        unlearnt_model,
        train_data,
        forget_data,
        retain_data,
        test_data,
    ):
        super().__init__(
            base_model, unlearnt_model, train_data, forget_data, retain_data, test_data
        )
        self.gmm_in = None
        self.gmm_out = None

    def compute_loss(self, model, dataset):
        model.model.eval()
        losses = np.array([])
        # criterion = nn.CrossEntropyLoss(reduction="none")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, targets = data
                inputs, targets = inputs.to(self.config["device"]), targets.to(
                    self.config["device"]
                )
                outputs = model.model(inputs)
                batch_losses = model.criterion(outputs, targets)

                losses = np.append(losses, [batch_losses.cpu().numpy()])
        return np.array(losses)

    def train_shadow_models(self, num_models=10):
        in_losses = []
        out_losses = []

        for n in range(num_models):
            indices = np.arange(len(self.retain_data))
            sampled_indices = np.random.choice(
                indices, size=5000, replace=False
            ).tolist()
            out_train_dataset = Subset(self.retain_data, sampled_indices)
            in_train_dataset = ConcatDataset([out_train_dataset, self.forget_data])
            in_train_loader = DataLoader(in_train_dataset, batch_size=64, shuffle=True)
            out_train_loader = DataLoader(
                out_train_dataset, batch_size=64, shuffle=True
            )

            # Train shadow model on the training subset
            model_in = TestModel(device=self.config["device"])

            if os.path.exists(f"./saved_models/in_shadow_model_{n}.pt"):
                model_in.load_model(f"./saved_models/in_shadow_model_{n}.pt")
            else:
                print("Training in shadow model ", n)
                model_in.train(train_loader=in_train_loader, num_epochs=15)
                model_in.save_model(f"./saved_models/in_shadow_model_{n}.pt")

            # Collect losses
            in_loss = self.compute_loss(model=model_in, dataset=in_train_dataset)

            # Train shadow model on the training subset
            model_out = TestModel(device=self.config["device"])
            if os.path.exists(f"./saved_models/out_shadow_model_{n}.pt"):
                model_out.load_model(f"./saved_models/out_shadow_model_{n}.pt")
            else:
                print("Training out shadow model ", n)
                model_out.train(train_loader=out_train_loader, num_epochs=15)
                model_out.save_model(f"./saved_models/out_shadow_model_{n}.pt")
            # Collect losses
            out_loss = self.compute_loss(model=model_out, dataset=out_train_dataset)

            in_losses.extend(in_loss)
            out_losses.extend(out_loss)

        return np.array(in_losses), np.array(out_losses)

    def learn_distributions(self):
        in_losses, out_losses = self.train_shadow_models()
        self.gmm_in = GaussianMixture(n_components=1).fit(in_losses.reshape(-1, 1))
        self.gmm_out = GaussianMixture(n_components=1).fit(out_losses.reshape(-1, 1))

    def evaluate_model(self, victim_model):
        target_losses = self.compute_loss(victim_model, self.forget_data)
        log_likelihood_in = self.gmm_in.score_samples(target_losses.reshape(-1, 1))
        log_likelihood_out = self.gmm_out.score_samples(target_losses.reshape(-1, 1))

        likelihood_ratios = log_likelihood_in - log_likelihood_out

        return likelihood_ratios

    def evaluate(self):
        if self.gmm_in is None or self.gmm_out is None:
            self.learn_distributions()

        threshold = 0.0

        # base_likelihood_ratios = self.evaluate_model(self.base_model)
        # base_membership_predictions = base_likelihood_ratios > threshold

        unlearnt_likelihood_ratios = self.evaluate_model(self.unlearnt_model)
        unlearnt_membership_predictions = unlearnt_likelihood_ratios > threshold

        # base_identified = np.sum(base_membership_predictions)
        unlearnt_identified = np.sum(unlearnt_membership_predictions)

        # print("Base Membership Predictions:", base_identified)
        print("Membership Predictions:", unlearnt_identified)


class ULiRAMIA(UnlearningMetric):
    def __init__(self):
        pass

    def evaluate(base_model, unlearnt_model, forget_data, retain_data):
        pass


class L2ErrorWeights(UnlearningMetric):
    def __init__(
        self,
        base_model,
        unlearnt_model,
        train_data,
        forget_data,
        retain_data,
        test_data,
    ):
        super().__init__(
            base_model, unlearnt_model, train_data, forget_data, retain_data, test_data
        )
        self.nr_model = TestModel(device=self.config["device"])
        self.nr_model.load_model("./saved_models/naive_retrained.pt")

    def evaluate(self):
        unlearnt_weights = self.unlearnt_model.model.state_dict()
        nr_weights = self.nr_model.model.state_dict()

        l2_error_unlearnt = 0

        for key in unlearnt_weights.keys():
            l2_error_unlearnt += torch.norm(
                unlearnt_weights[key].float() - nr_weights[key].float(), 2
            )

        print(
            "L2 Error between unlearnt and naive retrained model: ", l2_error_unlearnt
        )


class UnlearntInfluence(UnlearningMetric):
    def __init__(
        self,
        base_model,
        unlearnt_model,
        train_data,
        forget_data,
        retain_data,
        test_data,
    ):
        super().__init__(
            base_model, unlearnt_model, train_data, forget_data, retain_data, test_data
        )

    def hvp(self, loss, model, v):
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.contiguous().view(-1) for g in grad])
        grad_vector_product = torch.sum(grad_vector * v)
        hvp = torch.autograd.grad(grad_vector_product, model.parameters())
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def evaluate(self):
        self.unlearnt_model.model.eval()
        params = self.unlearnt_model.model.parameters()
        total_influence = 0
        for inputs, targets in self.forget_data:
            v = torch.randn(sum(p.numel() for p in params)).to(self.config["device"])
            inputs, targets = (
                inputs.to(self.config["device"]),
                torch.Tensor([targets]).to(self.config["device"]).int(),
            )
            outputs = self.unlearnt_model.model(inputs.unsqueeze(0))
            loss = self.unlearnt_model.criterion(outputs, targets)
            print("loss calculated")
            grad = torch.autograd.grad(loss, self.unlearnt_model.model.parameters())
            grad_vector = torch.cat([g.contiguous().view(-1) for g in grad])

            cur_estimate = v.clone()
            num_iterations = 50
            dampening = 0.01
            trainloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
            for _ in range(num_iterations):
                for train_inputs, train_targets in trainloader:
                    train_inputs, train_targets = train_inputs.to(
                        self.config["device"]
                    ), train_targets.to(self.config["device"])
                    train_outputs = self.unlearnt_model.model(train_inputs)
                    loss = self.unlearnt_model.criterion(train_outputs, train_targets)
                    hv = self.hvp(loss, self.unlearnt_model.model, cur_estimate)
                    cur_estimate = (
                        grad_vector
                        + (1 - dampening) * cur_estimate
                        - hv / len(trainloader.dataset)
                    )
            total_influence += cur_estimate.norm()
        print("Total Influence: ", total_influence)
        print("Average Influence: ", total_influence / len(self.forget_data))
