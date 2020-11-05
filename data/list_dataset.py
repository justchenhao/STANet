import os
import collections
import numpy as np
from data.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image


class listDataset(BaseDataset):
    """
    data structure
    -dataroot
        ├─A
            ├─train1.png
            ...
        ├─B
            ├─train1.png
            ...
        ├─label
            ├─train1.png
            ...
        └─list
            ├─val.txt
            ├─test.txt
            └─train.txt

    # In list/train.txt, each low writes the filename of each sample,
       # for example:
           list/train.txt
               train1.png
               train2.png
               ...
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.split = opt.split
        self.files = collections.defaultdict(list)
        self.istest = False if opt.phase == 'train' else True # 是否为测试/验证；若是，对数据不做尺度变换和旋转变换；

        path = os.path.join(self.root, 'list', self.split + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.split] = file_list
        # print(file_list)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        paths = self.files[self.split][index]

        path = paths.split(" ")

        A_path = os.path.join(self.root,'A', path[0])
        B_path = os.path.join(self.root,'B', path[0])

        L_path = os.path.join(self.root,'label', path[0])

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        tmp = np.array(Image.open(L_path), dtype=np.uint32) / 255
        # print(tmp.max())
        L = Image.fromarray(tmp)

        transform_params = get_params(self.opt, A.size, self.istest)

        transform = get_transform(self.opt, transform_params, test=self.istest)
        transform_L = get_transform(self.opt, transform_params, method=Image.NEAREST, normalize=False,
                                    test=self.istest)  # 标签不做归一化
        A = transform(A)
        B = transform(B)

        L = transform_L(L)

        return {'A': A, 'A_paths': A_path, 'B': B, 'B_paths': B_path,  'L': L, 'L_paths': L_path}


# Leave code for debugging purposes

if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    opt.dataroot = r'I:\data\change_detection\LEVIR-CD-r'
    opt.split = 'train'
    opt.load_size = 500
    opt.crop_size = 500
    opt.batch_size = 1
    opt.dataset_mode = 'list'
    from data import create_dataset
    dataset = create_dataset(opt)
    import matplotlib.pyplot as plt
    from util.util import tensor2im

    for i, data in enumerate(dataset):
        A = data['A']
        L = data['L']
        A = tensor2im(A)
        color = tensor2im(L)[:,:,0]*255
        plt.imshow(A)
        plt.show()

