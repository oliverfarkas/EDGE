import os
from pathlib import Path
import pickle

def fix_pickle(filepath):
    # Load the data from the pickle file
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize smpl_scaling if it's not already present
    if "smpl_scaling" not in data:
        data["smpl_scaling"] = []
    
    # Append scale values to smpl_scaling
    for a_trans in data["smpl_trans"]:
        scale_value = [1.0, 1.0, 1.0]  # Example transformation
        data["smpl_scaling"].append(scale_value)
    
    # Save the modified data back into a new pickle file
    modified_filepath = os.path.join(os.path.dirname(filepath) + "/", os.path.basename(filename))
    with open(modified_filepath, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    for root, dirs, files in os.walk("/Users/oliver/dev/EDGE/eval/motions/baseline.e51.b32.pt/"):
        for filename in files:
            file_path = Path(os.path.join(root, filename))
            if file_path.is_file() and file_path.suffix == '.pkl':
                fix_pickle(file_path)
        break