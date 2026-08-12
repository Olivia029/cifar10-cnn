import torchvision.transforms as transforms 
from torchvision.datasets import CIFAR10


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_datasets(data_dir="./data"):
    train_transform, test_transform = get_transforms()  
    
    train_dataset = CIFAR10(     
        root=data_dir,  
        train=True, 
        download=True,
        transform=train_transform,
    )
    
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset
