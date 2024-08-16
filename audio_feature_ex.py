import librosa as lr
import openl3
import numpy as np
import gc
import torch

JUKEBOX_SAMPLE_RATE = 44100
T = 8192
CTX_WINDOW_LENGTH = 5  # Example context window length in seconds
DEFAULT_DURATION = CTX_WINDOW_LENGTH
VQVAE_RATE = T / DEFAULT_DURATION

def empty_cache():
    gc.collect()

def load_audio(fpath, offset=0.0, duration=None):
    if duration is not None:
        audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset, duration=duration)
    else:
        audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset)

    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()

def get_z(audio):
    embeddings, _ = openl3.get_audio_embedding(audio, sr=JUKEBOX_SAMPLE_RATE, hop_size=1.0)
    return embeddings

def downsample(representation, target_rate=30, method="librosa_fft"):
    if method == "librosa_fft":
        resampled_reps = lr.resample(np.asfortranarray(representation.T), orig_sr=T / DEFAULT_DURATION, target_sr=target_rate).T
    return resampled_reps

@torch.no_grad()
def extract(audio=None, fpath=None, meanpool=False, layers=None, offset=0.0, duration=None, downsample_target_rate=None, downsample_method=None, force_empty_cache=True):
    if audio is None:
        assert fpath is not None
        if isinstance(fpath, list):
            audio = [load_audio(path, offset=offset, duration=duration) for path in fpath]
            bsize = len(fpath)
        else:
            audio = load_audio(fpath, offset=offset, duration=duration)
            bsize = 1
    elif fpath is None:
        assert audio is not None
        if isinstance(audio, list):
            bsize = len(audio)
        else:
            bsize = 1

    if force_empty_cache:
        empty_cache()

    z = get_z(audio)

    if force_empty_cache:
        empty_cache()

    acts = {36: z.squeeze()}  # Mimicking the layer output format

    if downsample_target_rate is not None:
        acts[36] = downsample(acts[36], target_rate=downsample_target_rate, method=downsample_method)

    if meanpool:
        acts = {num: act.mean(axis=0) for num, act in acts.items()}

    return acts

# Example usage
# Replace 'path_to_audio_file' with the actual path to the audio file you want to process
audio_features = extract(fpath='custom_music/01DropItLikeItsHot.wav', duration=10)
print(audio_features)
