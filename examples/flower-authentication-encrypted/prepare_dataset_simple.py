"""Prepare CIFAR-10 dataset partitions using torchvision."""

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset

DATASET_DIRECTORY = "datasets"


class CustomCIFAR10(Dataset):
    """Custom CIFAR-10 dataset wrapper."""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def save_dataset_to_disk(num_partitions: int = 2):
    """Download CIFAR-10 and create partitions."""
    
    print(f"Preparing {num_partitions} CIFAR-10 partitions...")
    
    # Create dataset directory
    os.makedirs(DATASET_DIRECTORY, exist_ok=True)
    
    # Download CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./tmp_cifar10', train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./tmp_cifar10', train=False, download=True, transform=transform
    )
    
    # Calculate partition sizes
    train_size = len(train_dataset) // num_partitions
    test_size = len(test_dataset) // num_partitions
    
    print(f"Train size per partition: {train_size}")
    print(f"Test size per partition: {test_size}")
    
    # Create partitions
    for i in range(num_partitions):
        print(f"Creating partition {i+1}/{num_partitions}...")
        
        # Create partition directory
        partition_dir = f"{DATASET_DIRECTORY}/cifar10_part_{i+1}"
        os.makedirs(partition_dir, exist_ok=True)
        
        # Get partition indices
        train_start = i * train_size
        train_end = (i + 1) * train_size if i < num_partitions - 1 else len(train_dataset)
        
        test_start = i * test_size  
        test_end = (i + 1) * test_size if i < num_partitions - 1 else len(test_dataset)
        
        # Extract partition data
        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, test_end))
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Save as torch tensors
        train_data = []
        train_targets = []
        for idx in train_indices:
            data, target = train_dataset[idx]
            train_data.append(data)
            train_targets.append(target)
        
        test_data = []
        test_targets = []
        for idx in test_indices:
            data, target = test_dataset[idx]
            test_data.append(data)
            test_targets.append(target)
        
        # Stack tensors
        train_data_tensor = torch.stack(train_data)
        train_targets_tensor = torch.tensor(train_targets)
        test_data_tensor = torch.stack(test_data)
        test_targets_tensor = torch.tensor(test_targets)
        
        # Save to disk
        torch.save({
            'train_data': train_data_tensor,
            'train_targets': train_targets_tensor,
            'test_data': test_data_tensor,
            'test_targets': test_targets_tensor
        }, f"{partition_dir}/data.pt")
        
        print(f"Partition {i+1} saved with {len(train_data)} train and {len(test_data)} test samples")
    
    print("Dataset preparation complete!")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree('./tmp_cifar10', ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save CIFAR-10 dataset partitions to disk")
    parser.add_argument(
        "num_partitions",
        type=int,
        nargs="?",
        default=2,
        help="Number of partitions to create (default: 2)",
    )
    
    args = parser.parse_args()
    save_dataset_to_disk(args.num_partitions)