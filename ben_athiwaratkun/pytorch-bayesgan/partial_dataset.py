from torch.utils.data import Dataset

class PartialDataset(Dataset):
    def __init__(self, full_dataset, num_points):
        self.num_points = num_points
        self.full_dataset = full_dataset
        # we're essentially restriting to the first 'num_points' datapoints

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.full_dataset.__getitem__(idx)

# BenA: tested. This can be used directly with dataset loader. The shuffling works. no 'memory leak'