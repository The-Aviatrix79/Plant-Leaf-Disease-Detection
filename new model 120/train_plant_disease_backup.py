import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the 'color' folder containing class subfolders
SOURCE_DATA_DIR = os.path.join("PlantVillage-Dataset-master", "raw", "color")
# Where the split dataset will be created
DEST_DATA_DIR = "dataset"
# Path to the external checkpoint (optional)
CHECKPOINT_PATH = "ResNet_101_ImageNet_plant-model-84.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS_HEAD = 5   # Epochs for training only the head
NUM_EPOCHS_FINE = 5   # Epochs for fine-tuning
LEARNING_RATE_HEAD = 0.001
LEARNING_RATE_FINE = 0.0001
SPLIT_RATIO = 0.8     # 80% train, 20% validation

# ==========================================
# 1. DATASET SPLITTING ONLY IF NOT EXISTS
# ==========================================
def split_dataset(source, dest, split_ratio):
    if os.path.exists(dest):
        print(f"Dataset folder '{dest}' already exists. Skipping split.")
        return

    print(f"Splitting dataset from '{source}' to '{dest}'...")
    train_dir = os.path.join(dest, "train")
    val_dir = os.path.join(dest, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    
    for class_name in classes:
        src_class_dir = os.path.join(source, class_name)
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create destination folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(train_dir, class_name, img))
        for img in val_images:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(val_dir, class_name, img))
            
    print("Dataset splitting complete.")

# ==========================================
# 2. DATA LOADERS & TRANSFORMS
# ==========================================
def get_dataloaders(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0) 
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                if i % 10 == 0:
                    print(f"  Batch {i}/{len(dataloaders[phase])}...", end='\r')
                    
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    print(f'Best val Acc: {best_acc:4f}')
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Prepare Data
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"Error: Source directory {SOURCE_DATA_DIR} not found.")
        exit(1)
        
    split_dataset(SOURCE_DATA_DIR, DEST_DATA_DIR, SPLIT_RATIO)
    dataloaders, dataset_sizes, class_names = get_dataloaders(DEST_DATA_DIR)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # 2. Initialize Model
    print("Initializing ResNet101...")
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # 3. Load Checkpoint (Strict=False)
    # This allows loading weights even if the final layer doesn't match yet,
    # or if we are about to replace it.
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            
            # Handle standard pytorch checkpoint vs state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out 'fc' keys because they have different shapes (20 classes vs 1000 or 38)
            # This prevents the size mismatch error and allows loading the backbone weights smoothly
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            
            # Load weights with strict=False to ignore mismatches (like the FC layer)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    else:
        print("Checkpoint path not found. Starting with ImageNet weights.")

    # 4. Modify Final Layer
    print(f"Replacing final layer for {num_classes} classes...")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # 5. Phase 1: Train Head Only (Freeze Backbone)
    print("\nPhase 1: Training Head (Backbone Frozen)...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer_head = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_HEAD)
    
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_head, num_epochs=NUM_EPOCHS_HEAD)

    # 6. Phase 2: Fine-tune (Unfreeze Layer4)
    print("\nPhase 2: Fine-tuning (Unfreezing Layer4)...")
    
    # Unfreeze layer4 and fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    # Lower learning rate for fine-tuning
    optimizer_fine = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE_FINE},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_HEAD} # Keep head LR or lower it too
    ], lr=LEARNING_RATE_FINE)
    
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_fine, num_epochs=NUM_EPOCHS_FINE)

    # 7. Final Inference Example
    print("\nPerforming inference on a sample validation image...")
    model.eval()
    
    # Get a batch
    inputs, labels = next(iter(dataloaders['val']))
    inputs = inputs.to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    # Show first result
    idx = 0
    print(f"Ground Truth: {class_names[labels[idx]]}")
    print(f"Prediction:   {class_names[preds[idx]]}")

    # Save final model
    save_path = "plant_disease_model_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
