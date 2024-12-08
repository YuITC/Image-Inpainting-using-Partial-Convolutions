from glob import glob
from PIL import Image, UnidentifiedImageError
import numpy as np

import torch
from torch.utils.data import Dataset

class InitDataset(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train', maskGenerator=None):
        super().__init__()
        self.img_transform  = img_transform
        self.mask_transform = mask_transform
        self.maskGenerator  = maskGenerator 
        self.paths          = glob(f'{data_root}/{data}/*')
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.load_img(self.paths[index])
        img = self.img_transform(img.convert('RGB'))
        
        mask = self.maskGenerator.generate() * 255
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = self.mask_transform(mask.convert('RGB'))

        # (masked_image, mask, original_image)
        # In model: input, mask, gt
        return img * mask, mask, img

    def load_img(self, path):
        try:
            img = Image.open(path)
            return img
        except (FileNotFoundError, UnidentifiedImageError):
            raise FileNotFoundError(f"x Can't load image from: {path}")
        
# # Testing
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torchvision.transforms.functional import to_pil_image
# from MaskGenerator import MaskGenerator

# dataset_val = InitDataset(
#     'E:/2-AIO/2-Project/Image-Inpainting/data/COCO2017',
#     transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]), 
#     transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]), 
#     data='val',
#     maskGenerator = MaskGenerator(256, 256, channels=3)
# )
# _, axes = plt.subplots(1, 3, figsize=(20, 4))
# for ax, ele in zip(axes, dataset_val[0]):
#     ax.imshow(to_pil_image(ele))
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.title(f"Dataset includes {len(dataset_val)}")
# plt.show()