import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import os
from util.util import save_visuals
from util.metrics import AverageMeter
import numpy as np
from util.util import mkdir

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


def print_current_acc(log_name, epoch, score):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """

    message = '(epoch: %s) ' % str(epoch)
    for k, v in score.items():
        message += '%s: %.3f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def val(opt):

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    save_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    mkdir(save_path)
    model.eval()
    # create a logging file to store training losses
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val1_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        score = model.test(val=True)           # run inference return confusion_matrix
        running_metrics.update(score)

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_visuals(visuals,save_path,img_path[0])
    score = running_metrics.get_scores()
    print_current_acc(log_name, opt.epoch, score)


if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    opt = make_val_opt(opt)
    opt.phase = 'val'
    opt.dataroot = 'path-to-LEVIR-CD-test'

    opt.dataset_mode = 'changedetection'

    opt.n_class = 2

    opt.SA_mode = 'PAM'
    opt.arch = 'mynet3'

    opt.model = 'CDFA'

    opt.name = 'LEVIR-CDFAp0'
    opt.results_dir = './results/'

    opt.epoch = '167_F1_1_0.89713'
    opt.num_test = np.inf

    val(opt)
