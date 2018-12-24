

from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
import tensorflow as tf
from models import Net2
import argparse
import params as hp

from data_load import Net2DataFlow


def get_eval_input_names():
    return ['x_mfccs', 'y_spec','y_mel']


def get_eval_output_names():
    return ['net2/eval/summ_loss']


def eval(logdir1, logdir2):
    # Load graph
    model = Net2()

    # dataflow
    df = Net2DataFlow(hp.Test2.data_path, hp.Test2.batch_size)

    ckpt1 = tf.train.latest_checkpoint(logdir1)
    ckpt2 = tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    if ckpt1:
        session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)

    x_mfccs, y_spec, y_mel = next(df().get_data())
    summ_loss, = predictor(x_mfccs, y_spec, y_mel)

    writer = tf.summary.FileWriter(logdir2)
    writer.add_summary(summ_loss)
    writer.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str,
                        help='experiment case name of train1')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case)

    eval(logdir1=logdir_train1, logdir2=logdir_train2)

    print("Done")
