import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# 1. Configuration
MODEL_PATH = 'ResNet_101_ImageNet_plant-model-84.pth'
NUM_CLASSES = 20  # Determined from model inspection

# IMPORTANT: The model file (.pth) does NOT contain the names of the diseases.
# It only outputs a number (0-19). You must provide the correct names in the list below
# or create a 'classes.txt' file in the same folder with one name per line.

# Attempt to load classes from a file, otherwise use placeholders
def load_class_names():
    class_file = 'classes.txt'
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            
            # If we have enough names, use them
            if len(names) >= NUM_CLASSES:
                print(f"Loaded {len(names)} classes from {class_file}")
                if len(names) > NUM_CLASSES:
                    print(f"Warning: Taking the first {NUM_CLASSES} classes.")
                    names = names[:NUM_CLASSES]
                
                # Clean up names (replace _ with space, remove triple underscores)
                clean_names = []
                for name in names:
                    clean = name.replace("___", " - ").replace("_", " ")
                    clean_names.append(clean)
                return clean_names
            else:
                print(f"Warning: {class_file} only has {len(names)} lines. Need {NUM_CLASSES}.")
    
    # Default variable names if file not found
    return [f"Class {i} (Unknown Disease)" for i in range(NUM_CLASSES)]

CLASS_NAMES = load_class_names()

# 2. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Load Model
def load_model(model_path, num_classes):
    print("Loading model architecture (ResNet101)...")
    # Initialize standard ResNet101
    model = models.resnet101(weights=None)
    
    # Modify the final layer (Fully Connected) to match the number of classes in the checkpoint
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"Loading weights from {model_path}...")
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle if checkpoint is a dictionary with 'state_dict' key or just the state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval() # Set to evaluation mode
        print("Model loaded successfully!")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return None

# 4. Define Image Transforms
# Standard transforms for ResNet: Resize to 256, CenterCrop to 224, Normalize
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Prediction Function
def predict_image(model, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = data_transforms(image).unsqueeze(0) # Add batch dimension
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        class_idx = predicted_idx.item()
        return CLASS_NAMES[class_idx], class_idx
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", -1

# 6. GUI Application
class PlantDiseaseApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Plant Disease Detector")
        self.root.geometry("600x500")
        
        # UI Elements
        self.label_title = tk.Label(root, text="Plant Disease Detection", font=("Helvetica", 16, "bold"))
        self.label_title.pack(pady=20)
        
        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Helvetica", 12), bg="#4CAF50", fg="white")
        self.btn_upload.pack(pady=10)
        
        self.label_image = tk.Label(root)
        self.label_image.pack(pady=10)
        
        self.label_result = tk.Label(root, text="Result: Waiting for image...", font=("Helvetica", 14))
        self.label_result.pack(pady=20)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            # Display image
            img = Image.open(file_path)
            img.thumbnail((300, 300)) # Resize for display
            photo = ImageTk.PhotoImage(img)
            self.label_image.config(image=photo)
            self.label_image.image = photo # Keep reference
            
            # Predict
            if self.model:
                prediction, idx = predict_image(self.model, file_path)
                
                # Determine color based on prediction content (heuristic)
                # Assuming "Healthy" might be in the name for good cases, else warning
                if "Healthy" in prediction or "healthy" in prediction:
                    color = "green"
                elif "Unknown" in prediction:
                    color = "orange"
                else:
                    color = "red"
                
                self.label_result.config(text=f"Prediction: {prediction}", fg=color)
                
                if "Unknown" in prediction:
                     messagebox.showinfo("Label Info", "The model predicted 'Class " + str(idx) + "', but the actual disease name is missing.\n\nPlease create a 'classes.txt' file with the 20 disease names or edit the code to define them.")

            else:
                self.label_result.config(text="Error: Model not loaded", fg="red")

# Main Execution
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        # Create a dummy window to show error
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Model file not found:\n{MODEL_PATH}")
    else:
        root = tk.Tk()
        loaded_model = load_model(MODEL_PATH, NUM_CLASSES)
        if loaded_model:
            app = PlantDiseaseApp(root, loaded_model)
            root.mainloop()
