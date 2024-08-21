import os
from args import parse_test_opt
from test import slice
import subprocess

from EDGE import EDGE

CHECKPOINT = "train-80.pt"
# CHECKPOINT = "EDGE.pt"

# Define the command and arguments
command = [
    '/Applications/Blender.app/Contents/MacOS/Blender', 
    '--background', 
    '--python', './render_from_pickle.py', 
    '--', 
    'eval/motions/' + CHECKPOINT + '/', 
    'renders/' + CHECKPOINT + '/'
]

def extract_features(opt):
    return slice(opt)

def generate_blender_data():
    result = subprocess.run(command, capture_output=True, text=True)


if __name__ == "__main__":
    # Adding the directory containing this script to sys.path
    os.environ['EDGE_CHECKPOINT'] = CHECKPOINT
    
    opt = parse_test_opt()
    data_tuples = extract_features(opt)

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    slice_up = False

    print("Generating dances")
    for data_tuple in data_tuples:
        if slice_up:
            start_idx = 0
            step = 2
            filenames = data_tuple[2]
            for file in filenames:
                # if (start_idx + step > len(filenames)):
                #     break
                print("Generating dance for " + file)
                target_dir = fk_out + os.path.basename(os.path.dirname(file)) + "/"
                model.render_sample(
                    data_tuple, os.environ['EDGE_CHECKPOINT'], opt.render_dir, start_idx, render_count=step, fk_out=target_dir, render=not opt.no_render
                )
                start_idx += 1
        else:
            model.render_sample(
                data_tuple, os.environ['EDGE_CHECKPOINT'], opt.render_dir, 0, render_count=-1, fk_out=fk_out, render=not opt.no_render
            )


    print("Done")
    # generate_blender_data()
