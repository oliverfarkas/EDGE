
import torch

# Load the state dictionary
state_dict = torch.load('/Users/oliver/.cache/jukemirlib/vqvae.pth.tar', map_location='mps', weights_only=True)

# Access the model parameters
model_state = state_dict['model']

# print(model_state)

# Open a file to write the results
with open('/Users/oliver/Desktop/output_channels_info.txt', 'w') as file:
    # Iterate through each parameter in the model state dictionary
    for param_name, param_tensor in model_state.items():
        # Print parameter name and its shape
        file.write(f"Parameter: {param_name}\n")
        file.write(f"Shape: {param_tensor.shape}\n")
        
        # Check if the parameter is a weight tensor of a convolutional layer
        if len(param_tensor.shape) == 4:  # Convolutional layer
            # Expected shape: (out_channels, in_channels, kernel_height, kernel_width)
            out_channels = param_tensor.shape[0]
            file.write(f"Output Channels: {out_channels}\n")
        elif len(param_tensor.shape) == 2:  # Linear layer
            # Expected shape: (out_features, in_features)
            out_channels = param_tensor.shape[0]
            file.write(f"Output Features: {out_channels}\n")
        else:
            file.write("Not a convolutional or linear layer parameter.\n")
        
        file.write("\n")  # Add a newline for better readability
