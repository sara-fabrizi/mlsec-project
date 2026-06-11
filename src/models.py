import torch.nn as nn
import torchvision.models as models

def get_resnet18(pretrained=False):
    """
    Initializes a ResNet-18 model adapted for the CIFAR-10 dataset.
    
    Args:
        pretrained (bool): If True, loads ImageNet pre-trained weights 
                           (useful as a starting point for fine-tuning).
    """
    # Select the updated parameter for weights if using recent torchvision,
    # otherwise maintain pretrained=True for backward compatibility.
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 1. Adaptation for 32x32 images (CIFAR-10)
    # Replace the initial 7x7 kernel (stride 2) with a 3x3 kernel (stride 1)
    # to preserve the initial spatial resolution.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Replace the initial maxpool with an Identity function to prevent further downsampling
    model.maxpool = nn.Identity()
    
    # 2. Adaptation of the final layer (Fully Connected layer)
    # ResNet-18 is originally designed for 1000 classes (ImageNet), but CIFAR-10 has only 10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model

if __name__ == '__main__':
    # Model integrity test
    print("Testing models module...")
    net = get_resnet18(pretrained=False)
    print(f"Final layer configured for {net.fc.out_features} classes.")
    print("CIFAR-10 modifications applied successfully to conv1 and maxpool.")