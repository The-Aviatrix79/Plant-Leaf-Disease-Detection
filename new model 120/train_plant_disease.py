import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import copy
import time
import json 
import warnings

# Use UserWarning for custom alerts but ignore minor torchvision warnings if desired
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# --- Configuration ---
# Assuming script is run from the directory containing 'dataset' and the checkpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
PRETRAINED_CHECKPOINT = os.path.join(BASE_DIR, 'ResNet_101_ImageNet_plant-model-84.pth')
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

BATCH_SIZE = 32
NUM_WORKERS = 4
# Check for CUDA availability and capability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU and will not run on CPU.")

try:
    # Test if the GPU is actually usable (kernels available)
    print(f"CUDA Available: True. Version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    test_tensor = torch.zeros(1).cuda()
    test_op = test_tensor * 2 
    DEVICE = torch.device("cuda")
    print("CUDA sanity check passed. Training on GPU.")
except RuntimeError as e:
    raise RuntimeError(f"CUDA is available but failed sanity check (likely missing kernels for this architecture). Error: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error checking CUDA. Error: {e}")

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Data Loading and Transforms ---
    print("Preparing data loaders...")
    
    # Standard ResNet transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)), # Resizing to 224x224
            transforms.RandomHorizontalFlip(), # Augmentation for training
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found at {TRAIN_DIR}")
        return

    # ImageFolder automatically sorts classes alphabetically
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    }

    # CRITICAL: Capture the class names exactly as ImageFolder sees them.
    # The index 0 corresponds to class_names[0], etc.
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Save class names to a JSON file for inspection/portability (Secondary precaution)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=4)
        print(f"Class names saved to {CLASS_NAMES_PATH}")

    # --- 2. Model Initialization ---
    print("Initializing model...")
    # Initialize ResNet101 with ImageNet weights
    model = models.resnet101(pretrained=True)

    # Load custom pretrained checkpoint if available
    if os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"Loading checkpoint from {PRETRAINED_CHECKPOINT}...")
        try:
            checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE)
            
            # Handle if checkpoint is a state_dict or a full model or nested
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()
            
            # strict=False allows loading even if the fc layer size doesn't match
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully (strict=False).")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Proceeding with standard ImageNet weights + partial custom weights.")
    else:
        print(f"Warning: Checkpoint {PRETRAINED_CHECKPOINT} not found. Using standard ImageNet weights.")

    # Modified final Fully Connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # --- 3. Phase 1: Train Head Only ---
    print("\n--- Phase 1: Training Classifier Head (Backbone Frozen) ---")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optimizer for head only
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=0.001)

    train_model(model, dataloaders, loss_fn, optimizer_phase1, num_epochs=5, phase_name="Head Training")

    # --- 4. Phase 2: Fine-Tuning ---
    print("\n--- Phase 2: Fine-Tuning (Unfreeze Layer4) ---")

    # Unfreeze Layer 4
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # FC is already unfrozen, but let's ensure it stays that way
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Use a smaller learning rate for fine-tuning
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer_phase2 = optim.Adam(params_to_update, lr=1e-4)

    train_model(model, dataloaders, loss_fn, optimizer_phase2, num_epochs=10, phase_name="Fine-Tuning")

    # --- 5. Save Model ---
    print(f"\nSaving model to {OUTPUT_MODEL_PATH}...")
    torch.save({
        'state_dict': model.state_dict(),
        'classes': class_names, # CRITICAL: Saving class names inside the model file
        'num_classes': num_classes
    }, OUTPUT_MODEL_PATH)
    print("Model saved.")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, phase_name="Training"):
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} [{phase_name}]')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'{phase_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


if __name__ == "__main__":
    main()
