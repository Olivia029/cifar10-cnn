from torch.utils.data import DataLoader

from dataset import get_datasets


def get_dataloaders(
    data_dir="./data", 
    train_batch_size=128,
    test_batch_size=100,
    num_workers=2, 
):
    train_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
