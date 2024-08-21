import gc
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from function import slice_waveform
from model import CleanUNet
from preprocess import common_preprocess


if __name__ == '__main__':
    with open('config/config.json') as f:
        data = f.read()
    config = json.loads(data)
    config_common = config['common']
    config_denoise = config['denoise']

    model = CleanUNet.load_from_checkpoint(config_denoise['model_path'])
    model.eval()

    os.makedirs(config_denoise['output_audio_path'], exist_ok=True)

    for file in os.listdir(config_denoise['noisy_audio_path']):
        if not file.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(config_denoise['noisy_audio_path'], file), sr=None)
        preprocessed = torch.tensor(common_preprocess(audio, sr, config_common['sampling_rate'])).unsqueeze(0)
        audio_length = preprocessed.size(1)

        slice_length = config_common['signal_length']

        sliced = slice_waveform(preprocessed, slice_length)
        waveforms = []
        for i in range(len(sliced)):
            with torch.no_grad():
                denoised = model(sliced[i].to(model.device)).squeeze(0)

                waveform_numpy = denoised.detach().cpu().numpy()

                if waveform_numpy.ndim == 2:
                    waveform_numpy = waveform_numpy[0]

                waveforms.append(waveform_numpy)

        final = np.concatenate(waveforms, axis=-1)[:audio_length]
        sf.write(os.path.join(config_denoise['output_audio_path'], file), final,
                 samplerate=config_common['sampling_rate'])
