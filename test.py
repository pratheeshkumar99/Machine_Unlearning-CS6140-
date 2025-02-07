import json
import torch
from models import TestModel
from mu_datasets import get_dataloader, get_dataset
from unlearning_algorithms import (
    NaiveRetraining,
    NegGrad,
    SCRUB,
    BadTeachingUnlearning,
)
from evaluations import (
    LiRAMIA,
    L2ErrorWeights,
    CaliberatedLossBasedClassifierMIA,
    UnlearntInfluence,
)

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(device)

    with open("config.json", "r") as f:
        config = json.load(f)
    config["device"] = device
    with open("config.json", "w") as f:
        json.dump(config, f)

    train_set = get_dataset(train=True)
    test_set = get_dataset(train=False)
    forget_set_test = get_dataset(forget=True, train_transforms=False)
    retain_set_test = get_dataset(retain=True, train_transforms=False)

    trainloader = get_dataloader(train=True)
    testloader = get_dataloader(train=False)
    retainloader = get_dataloader(retain=True)
    forgetloader = get_dataloader(forget=True)
    retainloader_test = get_dataloader(retain=True, train_transforms=False)
    forgetloader_test = get_dataloader(forget=True, train_transforms=False)

    model = TestModel(device)
    # model.train(trainloader, config["num_epochs_base"])
    model.load_model()
    print("Accuracy on training set: ")
    model.evaluate(trainloader)
    print("Accuracy on test set: ")
    model.evaluate(testloader)
    print("Accuracy on forget set: ")
    model.evaluate(forgetloader_test)
    print("Accuracy on retain set: ")
    model.evaluate(retainloader_test)
    # model.save_model()
    # liramia = LiRAMIA(
    #     model, model, forget_set_test, train_set, retain_set_test, test_set
    # )
    # liramia.evaluate()

    uinfluence = UnlearntInfluence(
        model, model, train_set, forget_set_test, retain_set_test, test_set
    )
    uinfluence.evaluate()

    nr = NaiveRetraining(
        model,
        model.optimizer,
        model.criterion,
        trainloader,
        retainloader,
        forgetloader,
        device,
    )
    # nr.unlearn()
    nr_model = nr.get_model()
    nr_model.load_model("./saved_models/naive_retrained.pt")
    print("\nNaive Retraining Model")
    print("Accuracy on training set: ")
    nr_model.evaluate(trainloader)
    print("Accuracy on test set: ")
    nr_model.evaluate(testloader)
    print("Accuracy on forget set: ")
    nr_model.evaluate(forgetloader_test)
    print("Accuracy on retain set: ")
    nr_model.evaluate(retainloader_test)
    # nr_model.save_model("./saved_models/naive_retrained.pt")

    # liramia = LiRAMIA(model, nr_model, forget_set_test, retain_set_test, test_set)
    # liramia.evaluate()
    model = TestModel(device)
    model.load_model()

    l2ew = L2ErrorWeights(
        model, nr_model, train_set, forget_set_test, retain_set_test, test_set
    )
    l2ew.evaluate()

    # clbmia = CaliberatedLossBasedClassifierMIA(
    #     model, nr_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # print(clbmia.evaluate())

    model = TestModel(device)
    model.load_model()

    negrad = NegGrad(
        model,
        model.optimizer,
        model.criterion,
        trainloader,
        retainloader,
        forgetloader,
        device,
    )
    negrad.unlearn()
    negrad_model = negrad.get_model()
    print("\nNegGrad Model")
    print("Accuracy on training set: ")
    negrad_model.evaluate(trainloader)
    print("Accuracy on test set: ")
    negrad_model.evaluate(testloader)
    print("Accuracy on forget set: ")
    negrad_model.evaluate(forgetloader_test)
    print("Accuracy on retain set: ")
    negrad_model.evaluate(retainloader_test)
    negrad_model.save_model("./saved_models/negrad_model.pt")

    model = TestModel(device)
    model.load_model()

    # liramia = LiRAMIA(
    #     model, negrad_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # liramia.evaluate()

    l2ew = L2ErrorWeights(
        model, negrad_model, train_set, forget_set_test, retain_set_test, test_set
    )
    l2ew.evaluate()

    # clbmia = CaliberatedLossBasedClassifierMIA(
    #     model, negrad_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # print(clbmia.evaluate())

    model = TestModel(device)
    model.load_model()

    scrub_mu = SCRUB(
        model,
        model.optimizer,
        model.criterion,
        trainloader,
        retainloader,
        forgetloader,
        device,
    )
    scrub_mu.unlearn()
    scrub_mu_model = scrub_mu.get_model()
    print("\nSCRUB Model")
    print("Accuracy on training set: ")
    scrub_mu_model.evaluate(trainloader)
    print("Accuracy on test set: ")
    scrub_mu_model.evaluate(testloader)
    print("Accuracy on forget set: ")
    scrub_mu_model.evaluate(forgetloader_test)
    print("Accuracy on retain set: ")
    scrub_mu_model.evaluate(retainloader_test)
    scrub_mu_model.save_model("./saved_models/scrub_mu_model.pt")

    model = TestModel(device)
    model.load_model()

    # liramia = LiRAMIA(
    #     model, scrub_mu_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # liramia.evaluate()

    l2ew = L2ErrorWeights(
        model, scrub_mu_model, train_set, forget_set_test, retain_set_test, test_set
    )
    l2ew.evaluate()

    # clbmia = CaliberatedLossBasedClassifierMIA(
    #     model, scrub_mu_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # print(clbmia.evaluate())

   

    # clbmia = CaliberatedLossBasedClassifierMIA(
    #     model, ncu_model, train_set, forget_set_test, retain_set_test, test_set
    # )
    # print(clbmia.evaluate())
    ### Bad-teaching-unlearning


    # Existing code
    model = TestModel(device)
    model.load_model()

    # Unlearning
    unlearning_teacher = TestModel(device).model
    full_trained_teacher = model.model
    un_model = BadTeachingUnlearning(
        model=model.model,
        optimizer=model.optimizer,
        criterion=model.criterion,
        train_loader=trainloader,
        retain_loader=retainloader_test,
        forget_loader=forgetloader_test,
        device=device,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=full_trained_teacher,
    )

    un_model.unlearn()
    un_model = TestModel(un_model.get_model())
    # # Assuming model is an instance of TestModel
    # model.save_model("saved_models/unlearned_model.pth")
    # print("Model state dictionary saved!")

    # Load the model using the custom TestModel class method
    # un_model = TestModel(device)
    # un_model.load_model("saved_models/unlearned_model.pth")

    # Evaluate using the TestModel's evaluate method
    print("Accuracy on training set: ")
    un_model.evaluate(trainloader)
    print("Accuracy on test set: ")
    un_model.evaluate(testloader)
    print("Accuracy on forget set: ")
    un_model.evaluate(forgetloader_test)
    print("Accuracy on retain set: ")
    un_model.evaluate(retainloader_test)
