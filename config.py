class Params:
    # the most important parameter
    self.random_state = 800008

    # system params
    self.verbose = True
    self.num_workers = 8
    self.device = None # to be set during runtime

    # wandb params
    self.use_wandb = False
    self.wandb_project = 'tacotron'
    self.num_examples = 5
    self.vocoder_file = 'vocoder.pt'

    # data location
    self.data_root = 'ljspeech/wavs/'
    self.metadata_file = 'ljspeech/metadata.csv'

    # checkpoints
    self.checkpoint_dir = 'checkpoints/'
    self.checkpoint_template = 'checkpoints/tacotron{}.pt'
    self.model_checkpoint = 'checkpoints/tacotron1.pt'
    self.load_model = False

    # data processing
    self.valid_ratio = 0.1
    self.max_audio_length = 222050
    self.max_chars_length = 187

    # melspectogramer params
    self.sample_rate = 22050
    self.win_length = 1024
    self.hop_length = 256
    self.n_fft = 1024
    self.f_min = 0
    self.f_max = 8000
    self.num_mels = 80
    self.power = 1.0
    self.pad_value = 1e-5

    # Tacotron params
    self.embed_dim = 512
    self.prenet_dim = 256
    self.attention_lstm_dim = 1024
    self.decoder_lstm_dim = 1024
    self.attention_dim = 128
    self.attention_temp = 0.08
    self.encoder_layers = 3
    self.kernel_size = 5
    self.postnet_layers = 3
    self.postnet_channels = 512
    self.attention_dropout = 0.1
    self.dropout = 0.5
    self.max_frames = 870
    self.threshold = 0.5

    # optimizer params
    self.lr = 3e-4
    self.weight_decay = 1e-4

    # training params
    self.start_epoch = 1
    self.num_epochs = 10


def set_params():
    return Params()

