import os
from functools import partial
from pathlib import Path

import librosa
import librosa as lr
import numpy as np
from tqdm import tqdm

import madmom
import matplotlib.pyplot as plt

FPS = 30
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6


def _get_tempo(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""

    # a lot of stuff, only take the 5th element
    audio_name = audio_name.split("_")[4]

    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    else:
        assert False, audio_name


import os
import numpy as np
from pathlib import Path
import librosa
from madmom.audio.signal import Signal
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def extract(fpath, skip_completed=True, dest_dir="aist_madmom_feats"):
    print("madmom extract: ")
    print(fpath)
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    # Load audio using librosa
    data, _ = librosa.load(fpath, sr=SR)

    # Feature extraction with librosa
    # envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
    ).T  # (seq_len, 12)

    signal = Signal("../" + str(fpath), sample_rate=SR)

    onset_processor = RNNOnsetProcessor(fps=FPS)
    envelope = onset_processor(signal) 

    # Onset detection using Madmom's RNNOnsetProcessor
    rnn_onset_processor = RNNOnsetProcessor()(signal)
    madmom_onset_processor = OnsetPeakPickingProcessor()
    madmom_onsets = madmom_onset_processor(rnn_onset_processor)
    madmom_onset_frames = np.round(madmom_onsets * FPS).astype(int)

    # Beat tracking using Madmom
    madmom_beat_processor = RNNBeatProcessor()(signal)
    madmom_beat_tracking_processor = DBNBeatTrackingProcessor(fps=100)
    madmom_beats = madmom_beat_tracking_processor(madmom_beat_processor)
    madmom_beats_frames = np.round(madmom_beats * FPS).astype(int)

    # Convert madmom features to one-hot encoding
    madmom_onset_onehot = np.zeros(151, dtype=np.float32)
    madmom_onset_onehot[madmom_onset_frames.astype(int)] = 1.0

    madmom_beat_onehot = np.zeros(151, dtype=np.float32)
    madmom_beat_onehot[madmom_beats_frames.astype(int)] = 1.0

    # # Create a time axis in seconds
    # time = np.linspace(0, len(signal) / signal.sample_rate * FPS, num=len(signal))

    # # Plot the signal
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, signal, label='Audio Signal')

    # # Add vertical lines for onsets (red) and beats (blue)
    # i = 0
    # for onset in madmom_onset_onehot:
    #     if onset == 1:
    #         plt.axvline(x=i, color='red', linestyle='--', label='madmom Onset' if i == np.where(madmom_onset_onehot == 1)[0][0] else "")
    #     i = i+1

    # i = 0
    # for beat in madmom_beat_onehot:
    #     if beat == 1:
    #         plt.axvline(x=i, color='green', linestyle='-', label='madmom beat' if i == np.where(madmom_beat_onehot == 1)[0][0] else "")
    #     i = i+1

    # plt.title('Audio Signal with Onsets and Beats')
    # plt.xlabel('Frames')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # Ensure madmom features match the librosa envelope length
    if len(madmom_onset_onehot) > len(envelope):
        madmom_onset_onehot = madmom_onset_onehot[:len(envelope)]
    if len(madmom_beat_onehot) > len(envelope):
        madmom_beat_onehot = madmom_beat_onehot[:len(envelope)]

    # Combine all features
    audio_feature = np.concatenate(
        [
            envelope[:, None], mfcc[:150], chroma[:150],
            madmom_onset_onehot[:, None], madmom_beat_onehot[:, None]
        ],
        axis=-1
    )

    # Chop to ensure exact shape
    audio_feature = audio_feature[:5 * FPS]
    assert (audio_feature.shape[0] - 5 * FPS) == 0, f"expected output to be ~5s, but was {audio_feature.shape[0] / FPS}"

    # Save or return the features
    # np.save(save_path, audio_feature)
    return audio_feature, save_path


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
