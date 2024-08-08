import torch
import torch.nn as nn
import torch.nn.functional as F

# Multiclass Dice Loss
class MultiClassDiceLoss(nn.Module):
    def __init__(self, labels=None):
        super(MultiClassDiceLoss, self).__init__()
        self.labels = labels 
        self.epsilon = torch.finfo(torch.float32).eps
        
    def forward(self, vol1, vol2, labels=None):
        """
        vol1: Tensor of shape (height, width) - predicted labels
        vol2: Tensor of shape (height, width) - ground truth labels
        labels: List of labels to calculate Dice coefficient for (excluding background)
        """
        labels = self.labels if labels is None else labels
        if labels is None:
            labels = torch.unique(torch.cat((vol1, vol2)))
            labels = labels[labels != 0]  # Remove background
        elif isinstance(labels, int):
            labels = [labels]
        
        dicem = torch.zeros(len(labels)).cuda()
        

        for idx, lab in enumerate(labels):
            vol1l = vol1 == lab
            vol2l = vol2 == lab
            top = 2 * torch.sum(vol1l & vol2l)
            bottom = torch.sum(vol1l) + torch.sum(vol2l)
            dicem[idx] = top / (bottom + self.epsilon)

        return 1 - dicem.mean()
        
if __name__ == "__main__":
    # Example usage
    a = torch.tensor([0,0,1,1,2,2,1,1,2,0])
    b = torch.tensor([0,1,1,1,0,2,1,1,2,0])

    # Create an instance of the MultiClassDiceLoss class
    criterion = MultiClassDiceLoss()

    # Compute the loss
    loss = criterion(a, b)
    print(loss)