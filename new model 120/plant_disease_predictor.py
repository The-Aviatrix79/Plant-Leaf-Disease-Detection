import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import json

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
CLASS_NAMES_JSON = os.path.join(BASE_DIR, 'class_names.json') # Optional fallback
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    """
    Simple Grad-CAM implementation for ResNet-like architectures.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates Grad-CAM heatmap for a specific class.
        Args:
           input_tensor: (1, C, H, W) input image
           class_idx: target class index (if None, uses predicted class)
        Returns:
           cam: (H, W) numpy array, normalized [0, 1]
           class_idx: efficient class index
           output: model raw output logits
        """
        # 1. Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # 2. Backward pass for the specific class
        # We want to maximize the output of the target class
        score = output[0, class_idx]
        score.backward()
        
        # 3. Compute Grad-CAM
        gradients = self.gradients # [1, 2048, 7, 7]
        activations = self.activations # [1, 2048, 7, 7]
        
        # Global Average Pooling of gradients to get importance weights
        # weights: [1, 2048] -> [2048]
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        # Weighted combination of feature maps
        # cam: [7, 7]
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=DEVICE)
        
        for i, w in enumerate(weights):
             cam += w * activations[0, i, :, :]
             
        # Apply ReLU to focus on features that have a positive influence on the class of interest
        cam = torch.maximum(cam, torch.tensor(0.0).to(DEVICE))
        
        # Normalize to 0-1 range
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.cpu().numpy(), class_idx, output

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # --- Critical: Load Class Names ---
    classes = None
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
        print(f"Loaded {len(classes)} classes from checkpoint dictionary.")
    elif os.path.exists(CLASS_NAMES_JSON):
        print(f"Checkpoint missing classes. Loading from {CLASS_NAMES_JSON}...")
        with open(CLASS_NAMES_JSON, 'r') as f:
            classes = json.load(f)
    else:
        raise ValueError("CRITICAL ERROR: No class names found in checkpoint or class_names.json. Cannot perform inference reliably without correct label mapping.")

    num_classes = len(classes)
    state_dict = checkpoint['state_dict']

    # --- Initialize Model ---
    model = models.resnet101(pretrained=False) # No need for ImageNet weights, we load full state
    # Replace last layer with correct number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # --- Defensive Check: Model Weights vs Num Classes ---
    # Check if the fc weights in state_dict match num_classes
    fc_weight = state_dict['fc.weight'] # [Out_Features, In_Features]
    if fc_weight.shape[0] != num_classes:
         raise RuntimeError(f"Mismatch! Model checkpoint has {fc_weight.shape[0]} output features, but {num_classes} class names were loaded. "
                            f"Classes: {classes}")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Ensure evaluation mode
    
    return model, classes

def predict_and_visualize(image_path, output_path=None):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # 1. Load and Transform Image
    # Transforms must match validation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]

    # 2. Load Model
    try:
        model, class_names = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # 3. Initialize Grad-CAM
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # 4. Run Inference
    heatmap, class_idx, output = grad_cam.generate_heatmap(input_tensor)
    
    # 5. Defensive Check: Index Bounds
    if class_idx < 0 or class_idx >= len(class_names):
        print(f"Error: Predicted index {class_idx} is out of bounds for {len(class_names)} classes.")
        return

    predicted_class = class_names[class_idx]
    # Calculate confidence
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence = probs[0][class_idx].item()
    
    print(f"Prediction: {predicted_class} (Index: {class_idx})")
    print(f"Confidence: {confidence:.2%}")

    # Safety: Threshold check
    CONFIDENCE_THRESHOLD = 0.75
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Warning: Confidence {confidence:.2%} is below threshold ({CONFIDENCE_THRESHOLD:.0%}). Grad-CAM may be unreliable and will not be generated.")
        return

    # 6. Visualization
    img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    
    # Optional: Threshold low values to reduce noise (e.g., allow only top 80% activations or absolute > 0.2)
    # heatmap[heatmap < 0.2] = 0 
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose with transparency
    # 0.6 * original + 0.4 * heatmap
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    result_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    if output_path is None:
        output_path = "prediction_result.jpg"
        
    cv2.imwrite(output_path, superimposed_img)
    print(f"Result saved to {output_path}")
    print("Note: Grad-CAM highlights regions of interest. It is not a segmentation mask.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", help="Path to save output", default="gradcam_result.jpg")
    
    args = parser.parse_args()
    
    predict_and_visualize(args.image_path, args.output)
