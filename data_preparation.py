import random
from collections import defaultdict

class PKsamplerWithLabels:
    """
    A P-K sampler that yields a tuple of (indices, labels) for each batch.
    
    NOTE: This sampler is intended for manual iteration, not for use with
    the DataLoader's `batch_sampler` argument.
    
    Args:
        labels (list): A list containing the label for each sample in the dataset.
        p (int): The number of classes per batch.
        k (int): The number of samples per class.
    """
    def __init__(self, labels, p, k):
        self.p = p
        self.k = k
        self.batch_size = self.p * self.k
        
        # Create a dictionary mapping each label to its indices
        self.labels_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.labels_to_indices[label].append(idx)
        
        self.unique_labels = list(self.labels_to_indices.keys())
        
        # Filter out classes with fewer than K samples
        self.valid_labels = [
            label for label in self.unique_labels 
            if len(self.labels_to_indices[label]) >= self.k
        ]

    def __iter__(self):
        # Shuffle the list of valid class labels for randomness between epochs
        random.shuffle(self.valid_labels)
        
        batch_indices = []
        batch_labels = [] 
        
        for label in self.valid_labels:
            indices = self.labels_to_indices[label]
            sampled_indices = random.sample(indices, self.k)
            
            # Add the indices and their corresponding labels to the batch lists
            batch_indices.extend(sampled_indices)
            batch_labels.extend([label] * self.k) 
            
            # If the batch is full, yield the data and reset
            if len(batch_indices) == self.batch_size:
                yield batch_indices, batch_labels
                batch_indices = []
                batch_labels = []

    def __len__(self):
        return len(self.valid_labels) // self.p