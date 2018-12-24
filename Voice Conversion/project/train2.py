

import argparse
import os
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.input_source.input_source import QueueInput
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SimpleTrainer
from tensorpack.utils import logger
import tensorflow as tf


from data_load import Net2DataFlow
import params as hp
from models import Net2
from utils import remove_all_files


def train(args, logdir1, logdir2):
    # model
    model = Net2()

    # dataflow
    df = Net2DataFlow(hp.Train2.data_path, hp.Train2.batch_size)

    # set logger for event and model saver
    logger.set_logger_dir(logdir2)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ), allow_soft_placement=True)

    session_inits = []
    ckpt2 = '{}/{}'.format(logdir2,
                           args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    ckpt1 = tf.train.latest_checkpoint(logdir1)
    if ckpt1:
        session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=5)),
        callbacks=[
            # TODO save on prefix net2
            ModelSaver(checkpoint_dir=logdir2),
            # ConvertCallback(logdir2, hp.Train2.test_per_epoch),
        ],
        max_epoch=hp.Train2.num_epochs,
        steps_per_epoch=hp.Train2.steps_per_epoch,
        session_init=ChainInit(session_inits),
        session_config=session_conf
    )
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    trainer = SimpleTrainer()

    launch_train_with_config(train_conf, trainer=trainer)


# def get_cyclic_lr(step):
#     lr_margin = hp.Train2.lr_cyclic_margin * math.sin(2. * math.pi / hp.Train2.lr_cyclic_steps * step)
#     lr = hp.Train2.lr + lr_margin
#     return lr


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str,
                        help='experiment case name of train1')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    #parser.add_argument('-r', action='store_true', help='start training from the beginning.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case)

    # if args.r:
    #    remove_all_files(logdir_train2)

    print('case: {}, logdir1: {}, logdir2: {}'.format(
        args.case, logdir_train1, logdir_train2))

    train(args, logdir1=logdir_train1, logdir2=logdir_train2)

    print("Done")
