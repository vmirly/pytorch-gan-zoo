from torchvision import datasets
from torch.utils.data import DataLoader


def get_loader(
        root_path,
        batch_size,
        transform,
        mode):

    if mode == 'train':
        is_train = shuffle = True
    else:
        is_train = shuffle = False

    dataset = datasets.MNIST(
        root=root_path,
        train=is_train,
        transform=transform,
        download=True)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle)

    return loader
