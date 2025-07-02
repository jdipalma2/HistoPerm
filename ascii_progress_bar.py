# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Progress Bars
=============

Use or override one of the progress bar callbacks.

"""
import importlib
import sys

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

from pytorch_lightning.callbacks.progress import ProgressBarBase


class ASCIIProgressBar(ProgressBarBase):
    """
    Updated the default progress bar to use ASCII values.
    Ubuntu screen command doesn't correctly print non-ASCII characters and results in unreadable output.
    Temporary workaround until screen command works correctly.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.test_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state['main_progress_bar'] = None
        state['val_progress_bar'] = None
        state['test_progress_bar'] = None
        return state

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            ascii=True
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True
        )
        return bar

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self.val_progress_bar = self.init_sanity_tqdm()
        self.val_progress_bar.total = convert_inf(sum(trainer.num_sanity_val_batches))
        self.main_progress_bar = tqdm(disable=True, ascii=True)  # dummy progress bar

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = self.init_train_tqdm()

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf'):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(convert_inf(total_batches))
        self.main_progress_bar.set_description(f'Epoch {trainer.current_epoch}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            self._update_bar(self.main_progress_bar)  # fill up remaining
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.total = convert_inf(self.total_val_batches)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.val_batch_idx, self.total_val_batches):
            self._update_bar(self.val_progress_bar)
            self._update_bar(self.main_progress_bar)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self.main_progress_bar.set_postfix(trainer.progress_bar_dict)
        self.val_progress_bar.close()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.main_progress_bar.close()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.test_batch_idx, self.total_test_batches):
            self._update_bar(self.test_progress_bar)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self.test_progress_bar.close()

    def _should_update(self, current, total):
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def _update_bar(self, bar):
        """ Updates the bar by the refresh rate without overshooting. """
        if bar.total is not None:
            delta = min(self.refresh_rate, bar.total - bar.n)
        else:
            # infinite / unknown size
            delta = self.refresh_rate
        if delta > 0:
            bar.update(delta)


def convert_inf(x):
    if x == float("inf"):
        return None
    return x
