import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='./LEVIR-CD', help='path to images (should have subfolders A, B, label)')
        parser.add_argument('--val_dataroot', type=str, default='./LEVIR-CD', help='path to images in the val phase (should have subfolders A, B, label)')

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='CDF0', help='chooses which model to use. [CDF0 | CDFA]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB ')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB')
        parser.add_argument('--arch', type=str, default='mynet3', help='feature extractor architecture | mynet3')
        parser.add_argument('--f_c', type=int, default=64, help='feature extractor channel num')
        parser.add_argument('--n_class', type=int, default=2, help='# of output pred channels: 2 for num of classes')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--SA_mode', type=str, default='BAM', help='choose self attention mode for change detection, | ori |1 | 2 |pyramid, ...')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='changedetection', help='chooses how datasets are loaded. [changedetection | concat | list | json]')
        parser.add_argument('--val_dataset_mode', type=str, default='changedetection', help='chooses how datasets are loaded. [changedetection | concat| list | json]')
        parser.add_argument('--dataset_type', type=str, default='CD_LEVIR', help='chooses which datasets too load. [LEVIR | WHU ]')
        parser.add_argument('--val_dataset_type', type=str, default='CD_LEVIR', help='chooses which datasets too load. [LEVIR | WHU ]')
        parser.add_argument('--split', type=str, default='train', help='chooses wihch list-file to open when use listDataset. [train | val | test]')
        parser.add_argument('--val_split', type=str, default='val', help='chooses wihch list-file to open when use listDataset. [train | val | test]')
        parser.add_argument('--json_name', type=str, default='train_val_test', help='input the json name which contain the file names of images of different phase')
        parser.add_argument('--val_json_name', type=str, default='train_val_test', help='input the json name which contain the file names of images of different phase')
        parser.add_argument('--ds', type=int, default='1', help='self attention module downsample rate')
        parser.add_argument('--angle', type=int, default=0, help='rotate angle')
        parser.add_argument('--istest', type=bool, default=False, help='True for the case without label')

        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | none]')
        parser.add_argument('--no_flip', type=bool, default=True, help='if specified, do not flip(left-right) the images for data augmentation')

        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        if dataset_name != 'concat':
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
