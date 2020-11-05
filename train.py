import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
from util import html
from util.visualizer import save_images
from util.metrics import AverageMeter
import copy
import numpy as np
import torch
import random


def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set seeds
# seed_torch(2019)
ifSaveImage = False

def make_val_opt(opt):

    val_opt = copy.deepcopy(opt)
    val_opt.preprocess = ''  #
    # hard-code some parameters for test
    val_opt.num_threads = 0   # test code only supports num_threads = 1
    val_opt.batch_size = 4    # test code only supports batch_size = 1
    val_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    val_opt.angle = 0
    val_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    val_opt.phase = 'val'
    val_opt.split = opt.val_split  # function in jsonDataset and ListDataset
    val_opt.isTrain = False
    val_opt.aspect_ratio = 1
    val_opt.results_dir = './results/'
    val_opt.dataroot = opt.val_dataroot
    val_opt.dataset_mode = opt.val_dataset_mode
    val_opt.dataset_type = opt.val_dataset_type
    val_opt.json_name = opt.val_json_name
    val_opt.eval = True

    val_opt.num_test = 2000
    return val_opt


def print_current_acc(log_name, epoch, score):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """
    message = '(epoch: %d) ' % epoch
    for k, v in score.items():
        message += '%s: %.3f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def val(opt, model):
    opt = make_val_opt(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers

    web_dir = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    model.eval()
    # create a logging file to store training losses
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        score = model.test(val=True)           # run inference
        running_metrics.update(score)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        if ifSaveImage:
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    score = running_metrics.get_scores()
    print_current_acc(log_name, epoch, score)
    if opt.display_id > 0:
        visualizer.plot_current_acc(epoch, float(epoch_iter) / dataset_size, score)
    webpage.save()  # save the HTML

    return score[metric_name]

metric_name = 'F1_1'


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    miou_best = 0
    n_epoch_bad = 0
    epoch_best = 0
    time_metric = AverageMeter()
    time_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'time_log.txt')
    with open(time_log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ training time (%s) ================\n' % now)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.train()
       # miou_current = val(opt, model)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            n_epoch = opt.niter + opt.niter_decay

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if ifSaveImage:
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            if total_iters % opt.print_freq == 0:   # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        t_epoch = time.time()-epoch_start_time
        time_metric.update(t_epoch)
        print_current_acc(time_log_name, epoch,{"current_t_epoch": t_epoch})


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            miou_current = val(opt, model)

            if miou_current > miou_best:
                miou_best = miou_current
                epoch_best = epoch
                model.save_networks(str(epoch_best)+"_"+metric_name+'_'+'%0.5f'% miou_best)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    time_ave = time_metric.average()
    print_current_acc(time_log_name, epoch, {"ave_t_epoch": time_ave})
