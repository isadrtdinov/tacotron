class Params:
    # the most important parameter
    random_seed = 808809

    # system params
    verbose = True
    num_workers = 8
    device = None # to be set during runtime
    vocoder_file = 'vocoder.pt'
    vocoder_dir = 'waveglow/'

    # wandb params
    use_wandb = True
    wandb_project = 'tacotron'

    # data location
    data_root = 'ljspeech/wavs/'
    metadata_file = 'ljspeech/metadata.csv'

    # checkpoints
    checkpoint_dir = 'checkpoints/'
    checkpoint_template = 'checkpoints/tacotron{}.pt'
    model_checkpoint = 'checkpoints/tacotron20.pt'
    load_model = True

    # data processing
    valid_ratio = 0.1
    max_audio_length = 222050
    max_chars_length = 187

    # melspectogramer params
    sample_rate = 22050
    win_length = 1024
    hop_length = 256
    n_fft = 1024
    f_min = 0
    f_max = 8000
    num_mels = 80
    power = 1.0
    pad_value = 1e-5

    # Tacotron params
    embed_dim = 512
    prenet_dim = 256
    attention_lstm_dim = 1024
    decoder_lstm_dim = 1024
    attention_dim = 128
    attention_temp = 0.08
    encoder_layers = 3
    kernel_size = 5
    postnet_layers = 3
    postnet_channels = 512
    attention_dropout = 0.1
    dropout = 0.5
    teacher_forcing = 0.5
    max_frames = 870
    threshold = 0.5
    frames_per_char = 5.75
    labels_temp = 0.08

    # optimizer params
    lr = 2e-4
    weight_decay = 1e-4

    # training params
    start_epoch = 21
    num_epochs = 10
    batch_size = 64


def set_params():
    return Params()

