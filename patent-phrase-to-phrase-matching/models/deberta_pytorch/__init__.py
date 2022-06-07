import torch
from utils.functions import KaggleFunctions as KF
from utils.config import Config
from utils.singleton import Singleton


class CFG(metaclass=Singleton):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    debug = True
    target_size = 1
    token_type_ids = False
    validation_size = 0.20
    max_len = 512
    model = Config(
        model_name='microsoft/deberta-v3-small',
        fc_dropout=0.2
    )
    batch_size = 32
    _epochs = 4

    steps_per_epoch = None
    pct_start = 0.3
    div_factor = 1e+2
    final_div_factor = 1e+4
    accumulate = 1
    seed = KF.seed_everything(42)

    # Train Behaviour
    num_workers = 4
    _folds = 4
    cv_monitor_value = "pearson"
    patience = 3

    optimizer = Config(
        name="AdamW",
        parameters=Config(
            lr=1e-5, weight_decay=0.01
        )
    )
    eps = 1e-6
    betas = (0.9, 0.999)
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    lr = 2e-5
    max_lr = 1e-3

    scheduler = 'cosine'  # ['linear', 'cosine']
    # Hyperparameters
    num_cycles = 0.5
    num_warmup_steps = 0

    scheduling_after = "step"
    pin_memory = True
    gradient_accumulation_steps = 1
    gradient_accumulation_batches = 1
    gradient_norm = 1.0
    gradient_scaling = True
    delta = 1e-4
    verbose = 100
    save_model = True
    amp = True

    apex = True
    max_grad_norm = 1000
    batch_scheduler = True

    @property
    def epochs(self):
        if self.debug:
            return 2
        return self._epochs

    @property
    def folds(self):
        if self.debug:
            return 1
        if self._folds <= 0:
            return 1
        return self._folds

    @property
    def fold_ids(self):
        if self.debug:
            return [0]
        return list(range(self.folds))
