import argparse
import datetime
import numpy as np
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.callbacks.base import Callback
from models import Net2
from audio import spec2wav, inv_preemphasis, db2amp, denormalize_db
from data_load import Net2DataFlow
import tensorflow as tf
import params as hp


# class ConvertCallback(Callback):
#     def __init__(self, logdir, test_per_epoch=1):
#         self.df = Net2DataFlow(hp.Convert.data_path, hp.Convert.batch_size)
#         self.logdir = logdir
#         self.test_per_epoch = test_per_epoch
#
#     def _setup_graph(self):
#         self.predictor = self.trainer.get_predictor(
#             get_eval_input_names(),
#             get_eval_output_names())
#
#     def _trigger_epoch(self):
#         if self.epoch_num % self.test_per_epoch == 0:
#             audio, y_audio, _ = convert(self.predictor, self.df)
#             # self.trainer.monitors.put_scalar('eval/accuracy', acc)
#
#             # Write the result
#             # tf.summary.audio('A', y_audio, hp.Default.sr, max_outputs=hp.Convert.batch_size)
#             # tf.summary.audio('B', audio, hp.Default.sr, max_outputs=hp.Convert.batch_size)


def convert(predictor, df):
    pred_spec, y_spec, ppgs = predictor(next(df().get_data()))

    # Denormalizatoin
    pred_spec = denormalize_db(pred_spec, hp.Default.max_db, hp.Default.min_db)
    y_spec = denormalize_db(y_spec, hp.Default.max_db, hp.Default.min_db)

    # Db to amp
    pred_spec = db2amp(pred_spec)
    y_spec = db2amp(y_spec)

    # Emphasize the magnitude
    pred_spec = np.power(pred_spec, hp.Convert.emphasis_magnitude)
    y_spec = np.power(y_spec, hp.Convert.emphasis_magnitude)

    # Spectrogram to waveform
    audio = np.array([spec2wav(spec.T, hp.Default.n_fft, hp.Default.win_length, hp.Default.hop_length,
                               hp.Default.n_iter) for spec in pred_spec])
    y_audio = np.array([spec2wav(spec.T, hp.Default.n_fft, hp.Default.win_length, hp.Default.hop_length,
                                 hp.Default.n_iter) for spec in y_spec])

    # Apply inverse pre-emphasis
    audio = inv_preemphasis(audio, coeff=hp.Default.preemphasis)
    y_audio = inv_preemphasis(y_audio, coeff=hp.Default.preemphasis)

    # if hp.Convert.one_full_wav:
    #     # Concatenate to a wav
    #     y_audio = np.reshape(y_audio, (1, y_audio.size), order='C')
    #     audio = np.reshape(audio, (1, audio.size), order='C')

    return audio, y_audio, ppgs


def get_eval_input_names():
    return ['x_mfccs', 'y_spec', 'y_mel']


def get_eval_output_names():
    return ['pred_spec', 'y_spec', 'ppgs']


def do_convert(args, logdir1, logdir2):
    # Load graph
    model = Net2()

    df = Net2DataFlow(hp.Convert.data_path, hp.Convert.batch_size)

    ckpt1 = tf.train.latest_checkpoint(logdir1)
    ckpt2 = '{}/{}'.format(logdir2,
                           args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
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

    audio, y_audio, ppgs = convert(predictor, df)

    # Write the result
    tf.summary.audio('A', y_audio, hp.Default.sr,
                     max_outputs=hp.Convert.batch_size)
    tf.summary.audio('B', audio, hp.Default.sr,
                     max_outputs=hp.Convert.batch_size)

    # Visualize PPGs
    heatmap = np.expand_dims(ppgs, 3)  # channel=1
    tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])

    writer = tf.summary.FileWriter(logdir2)
    with tf.Session() as sess:
        summ = sess.run(tf.summary.merge_all())
    writer.add_summary(summ)
    writer.close()

    # session_conf = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     device_count={'CPU': 1, 'GPU': 0},
    #     gpu_options=tf.GPUOptions(
    #         allow_growth=True,
    #         per_process_gpu_memory_fraction=0.6
    #     ),
    # )


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str,
                        help='experiment case name of train1')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    logdir_train1 = '{}/{}/train1'.format(hp.logdir_path, args.case)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case)

    print('case: {}, logdir1: {}, logdir2: {}'.format(
        args.case, logdir_train1, logdir_train2))

    s = datetime.datetime.now()

    do_convert(args, logdir1=logdir_train1, logdir2=logdir_train2)

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
