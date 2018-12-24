import argparse
import multiprocessing
import os

from tensorpack.callbacks.saver import ModelSaver
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
from tensorpack.utils import logger
from tensorpack.input_source.input_source import QueueInput
from data_load import Net1DataFlow
import params as hp
from models import Net1
import tensorflow as tf


def train(args, logdir):

    # model
    print("####model")
    model = Net1()

    # dataflow
    print("####dataflow")
    df = Net1DataFlow(hp.Train1.data_path, hp.Train1.batch_size)

    # set logger for event and model saver
    print("####logger")
    logger.set_logger_dir(logdir)

    print("####session_conf")
    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ), allow_soft_placement=True)

    print("####train_conf")
    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=1000, n_thread=5)),
        callbacks=[
            ModelSaver(checkpoint_dir=logdir),
            # TODO EvalCallback()
        ],
        max_epoch=hp.Train1.num_epochs,
        steps_per_epoch=hp.Train1.steps_per_epoch,
        session_config=session_conf
    )
    print("####ckpt")
    ckpt = '{}/{}'.format(logdir,
                          args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir)
    if ckpt:
        train_conf.session_init = SaverRestore(ckpt)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    print("####trainer")
    trainer = SyncMultiGPUTrainerReplicated(hp.Train1.num_gpu)

    print("####launch_train_with_config")
    launch_train_with_config(train_conf, trainer=trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str, help='experiment case name')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case)

    print('case: {}, logdir: {}'.format(args.case, logdir_train1))

    train(args, logdir=logdir_train1)

    print("Done")
