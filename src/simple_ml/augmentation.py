
from torchvision.transforms import transforms as tfs


def crop_flip_erase(h=32, w=32, crop_size=4):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    erase = tfs.RandomErasing(p=1., scale=(1 / 16, 1 / 16), ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])