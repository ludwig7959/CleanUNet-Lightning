import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import lightning as L
from torch.utils.data import Dataset, DataLoader

with open('config/config.json') as f:
    data = f.read()
config = json.loads(data)


class WaveformDataset(Dataset):
    def __init__(self, input_path, target_path):
        super().__init__()

        self.signal_length = config['common']['signal_length']

        self.inputs = []
        self.targets = []
        for file in sorted(list(Path(input_path).rglob('*.wav'))):
            target_file = os.path.join(target_path, file.name)
            if not os.path.isfile(target_file):
                print(f'Skipping {file} because there is no matching target.')
                continue

            input_waveform, _ = torchaudio.load(file)
            target_waveform, _ = torchaudio.load(target_file)

            input_waveform = self._cut(input_waveform)
            target_waveform = self._cut(target_waveform)

            self.inputs.append(input_waveform)
            self.targets.append(target_waveform)

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

        self.len_ = len(self.inputs)

        self.max = torch.max(torch.abs(self.inputs).max(), torch.abs(self.targets).max())

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def _cut(self, waveform):
        end = min(self.signal_length, waveform.shape[1])
        cut_waveform = waveform[:, 0:end]
        if cut_waveform.shape[1] < self.signal_length:
            cut_waveform = np.pad(cut_waveform, ((0, 0), (0, self.signal_length - cut_waveform.shape[1])), mode='constant', constant_values=0)

        return cut_waveform


class WaveformDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.use_validation = config['train']['validation']['enabled']

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = WaveformDataset(config['train']['input_audio_path'],
                                                 config['train']['target_audio_path'])
            if self.use_validation:
                self.val_dataset = WaveformDataset(config['train']['validation']['input_audio_path'],
                                                 config['train']['validation']['target_audio_path'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        if self.use_validation:
            return DataLoader(self.val_dataset,
                              batch_size=self.batch_size * 2,
                              num_workers=8,
                              persistent_workers=True
                              )
        return None

    def test_dataloader(self):
        return None
