from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import os
import numpy as np


class ChangeDetectionDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
            ├─A
            ├─B
            ├─label
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        folder_A = 'A'
        folder_B = 'B'
        folder_L = 'label'
        self.istest = False
        if opt.phase == 'test':
            self.istest = True
        self.A_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_A), opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_B), opt.max_dataset_size))
        if not self.istest:
            self.L_paths = sorted(make_dataset(os.path.join(opt.dataroot, folder_L), opt.max_dataset_size))

        print(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, A_img.size, test=self.istest)
        # apply the same transform to A B L
        transform = get_transform(self.opt, transform_params, test=self.istest)

        A = transform(A_img)
        B = transform(B_img)

        if self.istest:
            return {'A': A, 'A_paths': A_path, 'B': B, 'B_paths': B_path}

        L_path = self.L_paths[index]
        tmp = np.array(Image.open(L_path), dtype=np.uint32)/255
        L_img = Image.fromarray(tmp)
        transform_L = get_transform(self.opt, transform_params, method=Image.NEAREST, normalize=False,
                                    test=self.istest)
        L = transform_L(L_img)


        return {'A': A, 'A_paths': A_path,
                'B': B, 'B_paths': B_path,
                'L': L, 'L_paths': L_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
