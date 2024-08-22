import json
import os
import random

import librosa
import numpy as np

import soundfile as sf


def common_preprocess(audio, orig_sr, target_sr, target_rms=0.1):
    audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    audio_normalized = rms_normalize(audio_resampled, target_rms)

    return audio_normalized


def calculate_noise_rms(base_rms, snr):
    rms_noise = base_rms * 10 ** (-snr / 20)
    return rms_noise


def rms_normalize(y, target_rms):
    rms = np.sqrt(np.mean(y ** 2))
    scaling_factor = target_rms / rms
    normalized_y = y * scaling_factor

    return normalized_y


if __name__ == '__main__':
    with open('config/config.json') as f:
        data = f.read()
    config = json.loads(data)
    config_common = config['common']
    config_preprocess = config['preprocess']

    clean_audio_path = config_preprocess['clean_audio_path']
    noise_audio_path = config_preprocess['noise_audio_path']

    output_noisy_path = config_preprocess['output_noisy_path']
    output_clean_path = config_preprocess['output_clean_path']

    if not os.path.isdir(clean_audio_path):
        print(f'Directory {clean_audio_path} doesn''t exist.')
        exit(0)

    if not os.path.isdir(noise_audio_path):
        print(f'Directory {noise_audio_path} doesn''t exist.')
        exit(0)

    os.makedirs(output_noisy_path, exist_ok=True)
    os.makedirs(output_clean_path, exist_ok=True)

    # target_samples = int(TIME_DOMAIN_SIZE * HOP_LENGTH - HOP_LENGTH + 2)
    target_samples = config_common['signal_length']
    target_sr = config_common['sampling_rate']
    base_rms = config_common['base_rms']

    clean_audios = []
    for audio_file_name in os.listdir(clean_audio_path):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(clean_audio_path, audio_file_name), sr=None, mono=True)
        preprocessed = common_preprocess(audio, sr, target_sr, base_rms)

        end = min(target_samples, preprocessed.shape[0])
        cut_waveform = preprocessed[:end]
        if cut_waveform.shape[0] < target_samples:
            cut_waveform = np.pad(cut_waveform, (0, target_samples - cut_waveform.shape[0]), mode='constant', constant_values=0)
        clean_audios.append(cut_waveform)

    print('Number of clean audios: ', len(clean_audios))

    noise_audios = []
    for audio_file_name in os.listdir(noise_audio_path):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(noise_audio_path, audio_file_name), sr=None, mono=True)
        preprocessed = common_preprocess(audio, sr, target_sr, base_rms)
        audio_length = len(preprocessed)

        n = int(audio_length / target_samples)
        pad_length = target_samples * (n + 1) - audio_length
        current_length = 0
        audio_to_pad = np.array([], dtype=np.float32)
        while current_length < pad_length:
            clip_length = min(len(audio), pad_length - current_length)
            audio_to_pad = np.concatenate((audio_to_pad, audio[:clip_length]))
            current_length += clip_length
        padded_audio = np.concatenate((audio, audio_to_pad))

        for j in range(n + 1):
            start = target_samples * j
            noise_audios.append(padded_audio[start:start + target_samples])

    print('Number of noise audios: ', len(noise_audios))

    index = 0
    for snr in config_preprocess['signal_to_noise_ratios']:
        random.shuffle(clean_audios)
        target_rms = calculate_noise_rms(base_rms, snr)
        if len(clean_audios) > len(noise_audios):
            for i in range(len(clean_audios)):
                noise = rms_normalize(noise_audios[i % len(noise_audios)], target_rms)
                synthesized = clean_audios[i] + noise
                sf.write(os.path.join(output_noisy_path, f'audio-{index + 1}.wav'), synthesized,
                         samplerate=target_sr)
                sf.write(os.path.join(output_clean_path, f'audio-{index + 1}.wav'), clean_audios[i],
                         samplerate=target_sr)

                index += 1
        else:
            for i in range(len(noise_audios)):
                noise = rms_normalize(noise_audios[i], target_rms)
                synthesized = clean_audios[i % len(clean_audios)] + noise
                sf.write(os.path.join(output_noisy_path, f'audio-{index + 1}.wav'), synthesized,
                         samplerate=target_sr)
                sf.write(os.path.join(output_clean_path, f'audio-{index + 1}.wav'),
                         clean_audios[i % len(clean_audios)], samplerate=target_sr)

                index += 1
