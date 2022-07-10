import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


def is_image_file(filename):
    '''
        filter out non-image files
    '''
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def trans_to_tensor(pic):
    '''
        transform from image object to tensor
    '''
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float().div(255)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1):
    '''
        decide which augmentation method to apply
    '''
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise
    a = random.random()
    if flip == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        pass



mean_std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)
])


def function_label(x):
    if x > 0:
        return 1
    else:
        return 0


class RGBToGray(object):
    def __call__(self, mask):
        mask = mask.convert("L")
        mask = mask.point(function_label)
        return mask


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


target_transform = transforms.Compose([
    RGBToGray(),
    MaskToTensor()
])
palette = [0, 0, 0, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=320, size_h=320, flip=0):
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
        except OSError:
            return None, None, None

        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)

        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = initial_image.transpose(Image.FLIP_LEFT_RIGHT)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = initial_image.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        initial_image = input_transform(initial_image)
        semantic_image = target_transform(semantic_image)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)

class validation_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=320, size_h=320, flip=0):
        super(validation_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
        except OSError:
            return None, None, None

        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)


        initial_image = input_transform(initial_image)
        semantic_image = target_transform(semantic_image)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)