

# path
# local
#data_path_base = './datasets'
logdir_path = 'cases'
case = 'out'
# remote
# data_path_base = '/data/private/vc/datasets'
# logdir_path = 'data/private/vc/logdir'


class Default:
    # signal processing
    sr = 16000
    frame_shift = 0.006  # seconds
    frame_length = 0.071  # seconds #71  
    hop_length = 96  # 80 samples.  This is dependent on the frame_shift.
    win_length = 1136 # 400 samples. This is dependent on the frame_length.
    n_fft = 1136
    preemphasis = 0.97
    n_mels = 90
    n_mfcc = 60
    n_iter = 60  # Number of inversion iterations
    duration = 2
    max_db = 40
    min_db = -50

    # model
    hidden_units = 256  # alias = E
    num_banks = 16
    num_highway_blocks = 4
    norm_type = 'ins'  # a normalizer function. value = bn, ln, ins, or None
    t = 1.0  # temperature
    dropout_rate = 0.2

    # train
    batch_size = 32


class Train1:
    # path
    #data_path = 'Data/Train/*.wav'
    data_path= 'datasets/timit/raw/TIMIT/TRAIN/*/*/*.WAV'

    # model
    hidden_units = 128  # alias = E
    num_banks = 8
    num_highway_blocks = 4
    norm_type = 'ins'  # a normalizer function. value = bn, ln, ins, or None
    t = 1.0  # temperature
    dropout_rate = 0.2

    # train
    batch_size = 20
    lr = 0.0003
    num_epochs = 1000
    steps_per_epoch = 100
    save_per_epoch = 2
    num_gpu = 1


class Train2:
    # path
    data_path = 'Data/Train/Target/*.wav'

    # model
    hidden_units = 256  # alias = E
    num_banks = 8
    num_highway_blocks = 8
    norm_type = 'ins'  # a normalizer function. value = bn, ln, ins, or None
    t = 1.0  # temperature
    dropout_rate = 0.2

    # train
    batch_size = 50
    lr = 0.0003
    lr_cyclic_margin = 0.
    lr_cyclic_steps = 5000
    clip_value_max = 3.
    clip_value_min = -3.
    clip_norm = 10
    num_epochs = 10000
    steps_per_epoch = 100
    save_per_epoch = 50
    test_per_epoch = 1
    num_gpu = 1


class Test1:
    # path
    data_path = 'datasets/timit/raw/TIMIT/TEST/*/*/*.WAV' 

    # test
    batch_size = 32


class Test2:
    data_path = 'Data/Train/Target/*.wav'
    # test
    batch_size = 3


class Convert:
    # pathD:\deepvoice\convertaudio
    data_path = 'convertaudio/*.wav'

    # convert
    one_full_wav = True
    batch_size = 1
    emphasis_magnitude = 1.2
