import os
import pathlib
from PIL import Image
from torch.utils.data import Dataset


class PairedImg2ImgDataset(Dataset):
    """A dataset for paired image-to-image translation"""

    def __init__(
            self,
            image_dir,
            transform,
            paired_transform=None,
            preload=False,
            ext='jpg',
            mode='train'):
        """
        """
        self.paired_transform = paired_transform
        self.transform = transform
        self.preload = preload
        self.mode = mode

        image_dir = pathlib.Path(image_dir)

        self.files = [f for f in image_dir.glob('*.{}'.format(ext))]
        print('Number of images:', len(self.files))
        print(self.files[:4])

        if self.preload:
            self.images = [Image.open(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        if self.preload:
            img = self.images[index]
        else:
            img = Image.open(filename)

        img_width, img_height = img.size
        img_domain_a = img.crop([0, 0, img_width//2, img_height])
        img_domain_b = img.crop([img_width//2, 0, img_width, img_height])

        if self.paired_transform:
            img_domain_a, img_domain_b = self.paired_transform(
                (img_domain_a, img_domain_b))

        x = self.transform(img_domain_a)
        y = self.transform(img_domain_b)

        if self.mode == 'train':
            return x, y
        else:
            return x, y, os.path.basename(filename)
