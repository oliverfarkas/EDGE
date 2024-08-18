import os
from args import parse_test_opt
from test import test
import subprocess

# CHECKPOINT = "baseline.e10.b32.pt"
CHECKPOINT = "EDGE.pt"

# Define the command and arguments
command = [
    '/Applications/Blender.app/Contents/MacOS/Blender', 
    '--background', 
    '--python', './render_from_pickle.py', 
    '--', 
    'eval/motions/' + CHECKPOINT + '/', 
    'renders/' + CHECKPOINT + '/'
]

def extract_features():
    opt = parse_test_opt()
    test(opt)

def generate_blender_data():
    result = subprocess.run(command, capture_output=True, text=True)


if __name__ == "__main__":
    # Adding the directory containing this script to sys.path
    os.environ['EDGE_CHECKPOINT'] = CHECKPOINT
    
    extract_features()
    # generate_blender_data()
