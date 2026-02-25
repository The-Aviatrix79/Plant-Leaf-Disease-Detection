import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import json

# --- Configuration ---
# You may need to adjust this path if you move the app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
CLASS_NAMES_JSON = os.path.join(BASE_DIR, 'class_names.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes & Functions ---

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
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        # Weighted combination
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=DEVICE)
        for i, w in enumerate(weights):
             cam += w * activations[0, i, :, :]
             
        cam = torch.maximum(cam, torch.tensor(0.0).to(DEVICE))
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.cpu().numpy(), class_idx, output

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # --- Safe Class Loading ---
    classes = None
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    elif os.path.exists(CLASS_NAMES_JSON):
        with open(CLASS_NAMES_JSON, 'r') as f:
            classes = json.load(f)
    
    if classes is None:
        st.error("Critical Error: No class names found in checkpoint or class_names.json.")
        return None, None
        
    num_classes = len(classes)
    state_dict = checkpoint['state_dict']

    model = models.resnet101(pretrained=False)
    # Recreate the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Defensive check
    fc_weights = state_dict['fc.weight']
    if fc_weights.shape[0] != num_classes:
        st.error(f"Model mismatch! Checkpoint has {fc_weights.shape[0]} classes but {num_classes} names were loaded.")
        return None, None
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, classes

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def overlay_heatmap(heatmap, original_image):
    # original_image is PIL
    img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# --- UI Layout ---

st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("🌿 Plant Disease Detection with AI")
st.markdown(f"""
Upload an image of a plant leaf to detect potential diseases. 
The model uses a ResNet-101 architecture trained on the PlantVillage dataset using **{DEVICE}**.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display original
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Plant"):
            with st.spinner("Analyzing image..."):
                model, class_names = load_model()
                
                if model:
                    # Preprocess
                    input_tensor = process_image(image)
                    
                    # Grad-CAM Setup
                    # Hook into the last layer of the ResNet backbone
                    target_layer = model.layer4[-1]
                    grad_cam = GradCAM(model, target_layer)
                    
                    # Inference
                    heatmap, class_idx, output_logits = grad_cam.generate_heatmap(input_tensor)
                    
                    if class_idx < 0 or class_idx >= len(class_names):
                        st.error(f"Predicted index {class_idx} is out of bounds.")
                    else:
                        predicted_class = class_names[class_idx]
                        confidence = torch.nn.functional.softmax(output_logits, dim=1)[0][class_idx].item()

                        # Result Display
                        with col2:
                            st.header("Analysis Results")
                            st.success(f"**Prediction:** {predicted_class}")
                            st.info(f"**Confidence:** {confidence:.2%}")
                            
                            st.subheader("Explanation (Grad-CAM)")
                            
                            CONFIDENCE_THRESHOLD = 0.75
                            if confidence < CONFIDENCE_THRESHOLD:
                                st.warning(f"Confidence ({confidence:.2%}) is below the threshold ({CONFIDENCE_THRESHOLD:.0%}). Grad-CAM heatmap visualization is suppressed to prevent misleading interpretations.")
                            else:
                                st.write("The heatmap highlights the regions the model focused on to make this prediction.")
                                overlay = overlay_heatmap(heatmap, image)
                                st.image(overlay, caption="Heatmap Overlay", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")
