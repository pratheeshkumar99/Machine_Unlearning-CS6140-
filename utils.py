import torch
from torch.utils.data import TensorDataset, Dataset
import torch.nn.functional as F


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y
        


def process_dataloader(dataloader, is_retain=True):
    """
    Processes a DataLoader to create a dataset where:
    - X is the images.
    - Y is set to 1 if is_retain=True, otherwise 0.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The input DataLoader.
        is_retain (bool): Flag to determine the value of Y.
    
    Returns:
        torch.utils.data.TensorDataset: Dataset with processed X and Y.
    """
    # Collect all images and labels
    all_images = []
    for images, _ in dataloader:
        all_images.append(images)
    
    # Concatenate all batches to form the complete dataset
    all_images = torch.cat(all_images, dim=0)
    
    # Create labels based on the is_retain flag
    label_value = 1 if is_retain else 0
    all_labels = torch.full((all_images.size(0),), label_value, dtype=torch.long)
    
    # Create a new dataset
    processed_dataset = TensorDataset(all_images, all_labels)
    
    return processed_dataset


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

