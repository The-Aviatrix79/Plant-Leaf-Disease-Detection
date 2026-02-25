import torch
import os

model_path = 'ResNet_101_ImageNet_plant-model-84.pth'

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

try:
    # Try loading on CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    print(f"Successfully loaded '{model_path}'")
    print(f"Type of checkpoint: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Keys in checkpoint:", checkpoint.keys())
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' key.")
        else:
            state_dict = checkpoint
            print("Assuming dictionary is the state_dict.")
            
        # Inspect keys to guess architecture and num_classes
        keys = list(state_dict.keys())
        print(f"Number of keys in state_dict: {len(keys)}")
        print("First 5 keys:", keys[:5])
        print("Last 5 keys:", keys[-5:])
        
        # Try to find the final layer weights to deduce class count
        # ResNet usually ends with 'fc.weight' and 'fc.bias'
        fc_weight_keys = [k for k in keys if 'fc.weight' in k]
        if fc_weight_keys:
            key = fc_weight_keys[0]
            weight_tensor = state_dict[key]
            print(f"Found final layer weight '{key}': shape {weight_tensor.shape}")
            print(f"Estimated number of classes: {weight_tensor.shape[0]}")
        else:
            print("Could not find standard ResNet 'fc.weight' key. It might use a different naming convention.")
            
    elif isinstance(checkpoint, torch.nn.Module):
        print("Checkpoint is a full model object.")
        # We can try to print the model architecture
        print(checkpoint)
    else:
        print("Unknown checkpoint format.")

except Exception as e:
    print(f"Error loading model: {e}")
