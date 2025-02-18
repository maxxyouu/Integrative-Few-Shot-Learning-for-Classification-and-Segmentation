import os

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBar

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from common.evaluation import AverageMeter
from common import utils

from pprint import PrettyPrinter

from typing import Optional, Union
import math

# if importlib.util.find_spec("ipywidgets") is not None:
#     from tqdm.auto import tqdm as _tqdm
# else:

from tqdm import tqdm as _tqdm

_PAD_SIZE = 5
class tqdm(_tqdm):
    """
    Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from flickering
    """

    @staticmethod
    def format_num(n) -> str:
        """Add additional padding to the formatted numbers"""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n

def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values. We have to convert it to None."""
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x

def reset(bar: tqdm, total: Optional[int] = None) -> None:
    """Resets the tqdm bar to 0 progress with a new total, unless it is disabled."""
    if not bar.disable:
        bar.reset(total=convert_inf(total))

class CustomProgressBar(ProgressBar):
    """
    Custom progress bar for seperated training and validation processes
    """
    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True):
        super(CustomProgressBar, self).__init__()

        self.global_progress = global_progress
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

        self.main_progress_bar = tqdm()
        self.val_progress_bar = tqdm()

    def on_train_epoch_start(self, trainer, pl_module):
        total_train_batches = self.total_train_batches
        total_batches = total_train_batches
        reset(self.main_progress_bar, total_batches)
        self.main_progress_bar.set_description(f"[trn] ep: {trainer.current_epoch:>3}")

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            reset(self.val_progress_bar, sum(trainer.num_sanity_val_batches))
        else:
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.set_description(f"[val] ep: {trainer.current_epoch:>3}")
            reset(self.val_progress_bar, self.total_val_batches)

    def on_validation_end(self, trainer, pl_module):
        self.val_progress_bar.close()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._should_update(self.val_batch_idx, convert_inf(self.total_val_batches)):
            self._update_bar(self.val_progress_bar)

    def on_test_start(self, trainer, pl_module):
        self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.set_description(f'[test] {pl_module.args.benchmark} | fold{pl_module.args.fold} ')
        reset(self.test_progress_bar, self.total_test_batches)


class MeterCallback(Callback):
    """
    A class that initiates classificaiton and segmentation metrics
    """
    def __init__(self, args):
        super(MeterCallback, self).__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        print(pl_module.learner)
        utils.print_param_count(pl_module)

        if not self.args.nowandb and not self.args.eval:
            trainer.logger.experiment.watch(pl_module)

    def on_test_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        utils.print_param_count(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'\n\n----- ep: {trainer.current_epoch:>3}-----')
        utils.fix_randseed(None)
        dataset = trainer.train_dataloader.dataset# .datasets
        pl_module.average_meter = AverageMeter(dataset, self.args.way)
        pl_module.train_mode()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._shared_eval_epoch_start(trainer.val_dataloaders.dataset, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self._shared_eval_epoch_start(trainer.test_dataloaders.dataset, pl_module)

    def _shared_eval_epoch_start(self, dataset, pl_module):
        utils.fix_randseed(0)
        pl_module.average_meter = AverageMeter(dataset, self.args.way)
        pl_module.eval()


class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save
    """
    def __init__(self, args):
        self.dirpath = os.path.join('logs', args.benchmark, f'fold{args.fold}', args.backbone, args.logpath)
        if not args.eval and not args.resume:
            assert not os.path.exists(self.dirpath), f'{self.dirpath} already exists'
        self.filename = 'best_model'
        self.way = args.way
        self.weak = args.weak
        self.monitor = 'val/er' if args.weak else 'val/miou'

        super(CustomCheckpoint, self).__init__(dirpath=self.dirpath,
                                               monitor=self.monitor,
                                               filename=self.filename,
                                               mode='max',
                                               verbose=True,
                                               save_last=True)
        # For evaluation, load best_model-v(k).cpkt where k is the max index
        if args.eval:
            self.modelpath = self.return_best_model_path(self.dirpath, self.filename)
            print('evaluating', self.modelpath)
        # For training, set the filename as best_model.ckpt
        # For resuming training, pytorch_lightning will automatically set the filename as best_model-v(k).ckpt
        else:
            self.modelpath = os.path.join(self.dirpath, self.filename + '.ckpt')
        self.lastmodelpath = os.path.join(self.dirpath, 'last.ckpt') if args.resume else None

    def return_best_model_path(self, dirpath, filename):
        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if filename in ckpt_file]
        vers.sort()
        # vers = ['best_model.ckpt'] or
        # vers = ['best_model-v1.ckpt', 'best_model-v2.ckpt', 'best_model.ckpt']
        best_model = vers[-1] if len(vers) == 1 else vers[-2]
        return os.path.join(self.dirpath, best_model)


class OnlineLogger(WandbLogger):
    """
    A wandb logger class that is customed with the experiment log path
    """
    def __init__(self, args):
        super(OnlineLogger, self).__init__(
            name=args.logpath,
            project=f'fscs-{args.benchmark}-{args.backbone}',
            group=f'fold{args.fold}',
            log_model=False,
        )
        self.experiment.config.update(args)
