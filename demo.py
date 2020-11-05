import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import os
from util.util import save_images
import numpy as np
from util.util import mkdir
from PIL import Image


def make_val_opt(opt):

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip2 = True    # no flip; comment this line if results on flipped images are needed.

    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'val'
    opt.preprocess = 'none1'
    opt.isTrain = False
    opt.aspect_ratio = 1
    opt.eval = True

    return opt



def val(opt):

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    save_path = opt.results_dir
    mkdir(save_path)
    model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        pred = model.test(val=False)           # run inference return pred

        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(pred, save_path, img_path)


def pred_image(data_root, results_dir):
    opt = TestOptions().parse()   # get training options
    opt = make_val_opt(opt)
    opt.phase = 'test'
    opt.dataset_mode = 'changedetection'
    opt.n_class = 2
    opt.SA_mode = 'PAM'
    opt.arch = 'mynet3'
    opt.model = 'CDFA'
    opt.epoch = 'pam'
    opt.num_test = np.inf
    opt.name = 'pam'
    opt.dataroot = data_root
    opt.results_dir = results_dir

    val(opt)


if __name__ == '__main__':
    # define the data_root and the results_dir
    # note:
    # data_root should have such structure:
    # ├─A
    # ├─B
    # A for before images
    # B for after images
    data_root = './samples'
    results_dir = './samples/output/'
    pred_image(data_root, results_dir)
