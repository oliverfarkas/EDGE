import os
from functools import partial
from pathlib import Path

import openl3
import soundfile as sf
import numpy as np
from tqdm import tqdm

FPS = 30

AUDIO_SAMPLE_RATE = 44100
T = 8192

def average_pooling(array, target_size):
    original_size = array.shape[1]
    
    # Calculate the ratio of the sizes
    factor = original_size / target_size
    
    # Create an array to hold the pooled results
    pooled_array = np.zeros((array.shape[0], target_size), dtype=np.float32)

    
    for i in range(target_size):
        start_idx = int(i * factor)
        end_idx = int((i + 1) * factor)
        
        # Ensure we handle any rounding issues
        if end_idx > original_size:
            end_idx = original_size
        
        # Pool the corresponding segment
        pooled_array[:, i] = np.mean(array[:, start_idx:end_idx], axis=1)
    
    return pooled_array

def extract(fpath, skip_completed=True, dest_dir="aist_juke_feats", model=None):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    audio, sr = sf.read(fpath)
    hop_size = 0.0304
    # hop_size = 0.1
    emb, ts = openl3.get_audio_embedding(audio, model=model, hop_size=hop_size, sr=sr)

    # average pooling
    emb = average_pooling(emb, 4800)
    # simple end cutoff
    # emb = emb[:, :4800]

    #np.save(save_path, reps[LAYER])
    # return reps[LAYER], save_path
    return emb, ts


def extract_folder(src, dest):
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        np.save(path, rep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")

    args = parser.parse_args()

    extract_folder(args.src, args.dest)
