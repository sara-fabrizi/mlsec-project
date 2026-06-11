import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# =====================================================================
# 1. Custom Dataset for the Trigger Set (Watermark)
# =====================================================================
class TriggerSetDataset(Dataset):
    """
    This dataset loads the abstract images used as the watermark from ../data/trigger_set.
    It assigns a completely random target label to each image
    to defend against back-propagation based attacks.
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # List of all image files in the directory
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(self.img_names) == 0:
            raise ValueError(f"No images found in {img_dir}. Please ensure the trigger set is present.")
            
        # Random label assignment (classes 0-9 to be compatible with CIFAR-10)
        # We fix a seed to ensure reproducibility (the watermark mapping must be constant)
        torch.manual_seed(42)
        self.labels = torch.randint(0, 10, (len(self.img_names),))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # Always convert to RGB for consistency with CIFAR-10
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# =====================================================================
# 2. Infinite Iterator for the Trigger Set
# =====================================================================
def get_infinite_iterator(dataloader):
    """
    Since the trigger set only has 100 images, it would be exhausted much earlier
    than the CIFAR-10 epoch (which has 50,000). This iterator automatically
    restarts from the beginning when the loader is empty.
    """
    while True:
        for batch in dataloader:
            yield batch

# =====================================================================
# 3. Main Function to Prepare Dataloaders
# =====================================================================
def get_dataloaders(trigger_dir='../data/trigger_set', total_batch_size=128, trigger_size=2):
    """
    Prepares the dataloaders for the FROMSCRATCH training.
    
    Args:
        trigger_dir: Path to the directory containing the 100 abstract images.
        total_batch_size: The final batch size passed to the network (default 128).
        trigger_size: The 'k' value from the paper (watermark imgs per batch, default 2).
        
    Returns:
        clean_loader: Loader for standard CIFAR-10 data.
        trigger_loader: Infinitely iterable loader for watermark images.
        test_loader: Loader for final validation on CIFAR-10.
    """
    # Calculate how many "clean" samples are needed to reach the total batch size
    clean_batch_size = total_batch_size - trigger_size

    # Standard transformations for CIFAR-10 (Training)
    transform_train_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Transformations for Testing (no data augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # --- 1. Dataloader for CIFAR-10 (Clean Data) ---
    clean_trainset = torchvision.datasets.CIFAR10(
        root='../data/cifar10', train=True, download=True, transform=transform_train_cifar
    )
    clean_loader = DataLoader(
        clean_trainset, batch_size=clean_batch_size, shuffle=True, num_workers=2
    )

    # --- 2. Dataloader for the Trigger Set ---
    trigger_trainset = TriggerSetDataset(
        img_dir=trigger_dir, transform=transform_test
    )
    # drop_last=True is vital to ensure no batch has k < 2
    trigger_loader = DataLoader(
        trigger_trainset, batch_size=trigger_size, shuffle=True, drop_last=True
    )

    # --- 3. Dataloader for the Test Set (CIFAR-10) ---
    testset = torchvision.datasets.CIFAR10(
        root='../data/cifar10', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return clean_loader, trigger_loader, test_loader

if __name__ == '__main__':
    print("Testing data module...")
    os.makedirs('../data/trigger_set', exist_ok=True)
    print("\nWARNING: Remember to place 100 .png or .jpg images in './data/trigger_set' before running the real training.")