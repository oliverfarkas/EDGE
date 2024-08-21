import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import openl3
import numpy as np
import torch
from tqdm import tqdm
import librosa

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract
from data.audio_extraction.madmom_features import extract as madmom_extract
from pickle_fix import fix_pickle

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

def slice(opt):
    # Get the extraction function 
    feature_func = None
    if opt.feature_type == "jukebox":
        feature_func = juke_extract 
    elif opt.feature_type == "madmom":
        feature_func = madmom_extract
    else:
        feature_func = baseline_extract 

    # Get all precomputed features directories
    feature_dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))

    temp_dir_list = []
    all_cond = []
    all_filenames = []

    # Iterate all wav files in the music directory
    for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):

        y, sr = librosa.load(wav_file, sr=None)
        # Calculate the duration in seconds
        sample_length = librosa.get_duration(y=y, sr=sr)
        # sample_length = opt.out_length
        sample_size = int(sample_length / 2.5) - 1

        cond_list = []


        # Get the precomputed features directory name
        songname = os.path.splitext(os.path.basename(wav_file))[0]
        save_dir = os.path.join(opt.feature_cache_dir, songname)
        dirname = save_dir 

        file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
        juke_file_list = sorted(glob.glob(f"{dirname}/*.npy"), key=stringintkey)

        # Check if the features were already precomputed, and the num of slices equals to the num of numpys
        if (((dirname + '/') in feature_dir_list) and (len(file_list) == len(juke_file_list))):
            # rand_idx = random.randint(0, len(file_list) - sample_size)
            rand_idx = 0
            file_list = file_list[0 : sample_size]
            juke_file_list = juke_file_list[0 : sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
        else:
            # Create precomputed features directory
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            # Slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)

           # randomly sample a chunk of length at most sample_size
            rand_idx = 0
            # rand_idx = random.randint(0, len(file_list) - sample_size)
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                # if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    # continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                # if rand_idx <= idx < rand_idx + sample_size:
                cond_list.append(reps)

            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[0 : sample_size])  
  

        retval = []
        # print("Generating dances...")
        for i in range(len(all_cond)):
            data_tuple = None, all_cond[i], all_filenames[i]
            retval.append(data_tuple)
            # Open a file in write mode (this will create the file if it doesn't exist)
            # with open('output.txt', 'w') as file:
            #     # Loop through the lists and write the tuple to the file
            #     for i in range(len(all_cond)):
            #         data_tuple = (None, all_cond[i], all_filenames[i])
            #         # Write the tuple to the file, converting it to a string
            #         file.write(str(data_tuple) + '\n')
            # model.render_sample(
            #     data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
            # )
        print("Done")
        return retval
        
        # data_tuple = None, cond_list, file_list[0 : sample_size]
        # model.render_sample(
        #     data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        # )        

        # print("Generating dances")
        # for i in range(len(all_cond)):
        #     data_tuple = None, all_cond[i], all_filenames[i]
        #     model.render_sample(
        #         data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        #     )
        # print("Done")


# def test(opt):
#     feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract 

#     sample_length = opt.out_length
#     sample_size = int(sample_length / 2.5) - 1

#     temp_dir_list = []
#     all_cond = []
#     all_filenames = []
#     if opt.use_cached_features:
#         print("Using precomputed features")
#         # all subdirectories
#         dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
#         for dir in dir_list:
#             file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
#             juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
#             assert len(file_list) == len(juke_file_list)
#             # random chunk after sanity check
#             # rand_idx = random.randint(0, len(file_list) - sample_size)
#             rand_idx = 0
#             file_list = file_list[rand_idx : rand_idx + sample_size]
#             juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
#             cond_list = [np.load(x) for x in juke_file_list]
#             all_filenames.append(file_list)
#             all_cond.append(torch.from_numpy(np.array(cond_list)))
#     else:
#         print("Computing features for input music")
#         for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
#             # create temp folder (or use the cache folder if specified)
#             if opt.cache_features:
#                 songname = os.path.splitext(os.path.basename(wav_file))[0]
#                 save_dir = os.path.join(opt.feature_cache_dir, songname)
#                 Path(save_dir).mkdir(parents=True, exist_ok=True)
#                 dirname = save_dir
#             else:
#                 temp_dir = TemporaryDirectory()
#                 temp_dir_list.append(temp_dir)
#                 dirname = temp_dir.name
#             # slice the audio file
#             # sample_length = librosa.get_duration(filename=wav_file)
#             # sample_size = int(sample_length / 2.5) - 1
#             print(f"Slicing {wav_file}")
#             slice_audio(wav_file, 2.5, 5.0, dirname)
#             file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
#             # randomly sample a chunk of length at most sample_size
#             rand_idx = 0
#             # rand_idx = random.randint(0, len(file_list) - sample_size)
#             cond_list = []
#             # generate juke representations
#             print(f"Computing features for {wav_file}")
#             for idx, file in enumerate(tqdm(file_list)):
#                 # if not caching then only calculate for the interested range
#                 # if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
#                     # continue
#                 # audio = jukemirlib.load_audio(file)
#                 # reps = jukemirlib.extract(
#                 #     audio, layers=[66], downsample_target_rate=30
#                 # )[66]
#                 reps, _ = feature_func(file)
#                 # save reps
#                 if opt.cache_features:
#                     featurename = os.path.splitext(file)[0] + ".npy"
#                     np.save(featurename, reps)
#                 # if in the random range, put it into the list of reps we want
#                 # to actually use for generation
#                 # if rand_idx <= idx < rand_idx + sample_size:
#                 cond_list.append(reps)
#             cond_list = torch.from_numpy(np.array(cond_list))
#             all_cond.append(cond_list)
#             all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

#     model = EDGE(opt.feature_type, opt.checkpoint)
#     model.eval()

#     # directory for optionally saving the dances for eval
#     fk_out = None
#     if opt.save_motions:
#         fk_out = opt.motion_save_dir

#     print("Generating dances")
#     for i in range(len(all_cond)):
#         data_tuple = None, all_cond[i], all_filenames[i]
#         model.render_sample(
#             data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
#         )
#     print("Done")
#     torch.cuda.empty_cache()
#     for temp_dir in temp_dir_list:
#         temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
