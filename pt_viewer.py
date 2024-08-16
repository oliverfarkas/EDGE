import torch

# Load the .pt file
checkpoint = torch.load('checkpoint.pt', map_location=torch.device('mps'))

# Inspect the type of the checkpoint
print(type(checkpoint))
