import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Mammography AI Decision Support System",
    page_icon="🩺",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"C:\Users\thera\Mammography_Breast_Cancer_Project"

@st.cache_resource
def load_model(model_name):
    if model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
        path = os.path.join(BASE_DIR, "best_vgg16.pth")
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        path = os.path.join(BASE_DIR, "best_resnet50.pth")
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        path = os.path.join(BASE_DIR, "best_efficientnetb0.pth")
    else:
        raise ValueError("Unknown model")
        
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np)
    img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    return transform(pil_img).unsqueeze(0).to(DEVICE)

# App Header
st.title("🩺 Mammography Breast Cancer Detection System")
st.markdown("### Professional AI-Powered Clinical Decision Support Prototype")
st.info("⚠️ **Disclaimer:** This software is a college research prototype developed using transfer learning CNNs. It is intended for academic demonstration and decision-support analysis only, not as a standalone clinical diagnostic tool.")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Model Inspector (One-by-One)", 
    "⚖️ Multi-Model Side-by-Side Comparison", 
    "📊 Global Performance & Metrics"
])

# -----------------------------------------------------------------------------
# TAB 1: MODEL INSPECTOR (ONE BY ONE)
# -----------------------------------------------------------------------------
with tab1:
    st.header("Individual Model Deep Dive")
    st.markdown("Select a specific neural network architecture to inspect its configuration, test metrics, and run individual image inferences.")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        chosen_model = st.selectbox("Choose AI Architecture", ["VGG16", "ResNet50", "EfficientNetB0"], key="inspector_select")
        
        # Architecture descriptions
        arch_info = {
            "VGG16": "Deep sequential CNN architecture characterized by uniform $3 \\times 3$ convolutional filters. High parameter count, excellent at capturing micro-spatial patterns.",
            "ResNet50": "50-layer deep residual network utilizing skip connections to mitigate vanishing gradients. Balanced feature representation for mammographic masses.",
            "EfficientNetB0": "Lightweight, highly optimized architecture utilizing compound scaling and mobile inverted bottleneck blocks. High efficiency and fast inference speed."
        }
        st.success(arch_info[chosen_model])
        
    with col_sel2:
        st.subheader(f"Upload Mammogram for {chosen_model}")
        single_file = st.file_uploader("Upload test image (.jpg / .jpeg)", type=["jpg", "jpeg", "png"], key="single_upload")
        
        if single_file is not None:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                input_img = Image.open(single_file).convert("L")
                st.image(input_img, caption="Uploaded Mammogram", use_container_width=True)
                
            with col_img2:
                if st.button(f"Run {chosen_model} Analysis", type="primary", key="run_single"):
                    with st.spinner(f"Extracting features using {chosen_model}..."):
                        tensor_img = preprocess_image(input_img)
                        model = load_model(chosen_model)
                        
                        with torch.no_grad():
                            outputs = model(tensor_img)
                            probs = torch.softmax(outputs, dim=1)
                            pred_class = torch.argmax(probs, dim=1).item()
                            conf = probs[0][pred_class].item() * 100
                            
                        label_str = "Malignant (Class 1)" if pred_class == 1 else "Benign (Class 0)"
                        
                        if pred_class == 1:
                            st.error(f"**Prediction:** {label_str}")
                        else:
                            st.success(f"**Prediction:** {label_str}")
                            
                        st.metric(label="Confidence Score", value=f"{conf:.2f}%")
                        st.write(f"**Raw Probabilities:** Benign: {probs[0][0].item()*100:.1f}% | Malignant: {probs[0][1].item()*100:.1f}%")

# -----------------------------------------------------------------------------
# TAB 2: SIDE-BY-SIDE COMPARISON
# -----------------------------------------------------------------------------
with tab2:
    st.header("Multi-Model Comparative Inference")
    st.markdown("Upload a single mammogram image to evaluate it across **all three CNNs simultaneously** and observe how their predictions and confidence levels differ.")
    
    multi_file = st.file_uploader("Upload test mammogram...", type=["jpg", "jpeg", "png"], key="multi_upload")
    
    if multi_file is not None:
        multi_img = Image.open(multi_file).convert("L")
        st.image(multi_img, caption="Test Image for Multi-Model Evaluation", width=300)
        
        if st.button("Run All Models Simultaneously", type="primary"):
            tensor_img = preprocess_image(multi_img)
            
            col_vgg, col_res, col_eff = st.columns(3)
            
            models_list = [("VGG16", col_vgg), ("ResNet50", col_res), ("EfficientNetB0", col_eff)]
            
            for m_name, col in models_list:
                with col:
                    st.markdown(f"### {m_name}")
                    with st.spinner(f"Running {m_name}..."):
                        model = load_model(m_name)
                        with torch.no_grad():
                            outputs = model(tensor_img)
                            probs = torch.softmax(outputs, dim=1)
                            pred_class = torch.argmax(probs, dim=1).item()
                            conf = probs[0][pred_class].item() * 100
                            
                        lbl = "Malignant" if pred_class == 1 else "Benign"
                        if pred_class == 1:
                            st.error(f"**{lbl}**")
                        else:
                            st.success(f"**{lbl}**")
                            
                        st.metric("Confidence", f"{conf:.2f}%")
                        st.write(f"P(Benign): {probs[0][0].item()*100:.1f}%")
                        st.write(f"P(Malignant): {probs[0][1].item()*100:.1f}%")

# -----------------------------------------------------------------------------
# TAB 3: GLOBAL PERFORMANCE & METRICS
# -----------------------------------------------------------------------------
with tab3:
    st.header("Global Test Evaluation Metrics")
    st.markdown("Performance benchmarks evaluated across the independent test dataset ($N = 452$ patient cases).")
    
    st.subheader("Model Comparison Benchmark Table")
    st.markdown("""
    | Model Architecture | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | AUC-ROC |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **VGG16** | 60.40% | 0.00% | 100.00% | 0.00% | 0.00% | 0.489 |
    | **ResNet50** | 60.40% | 0.00% | 100.00% | 0.00% | 0.00% | 0.489 |
    | **EfficientNetB0** | 60.40% | 0.00% | 100.00% | 0.00% | 0.00% | 0.489 |
    """)
    
    st.subheader("Receiver Operating Characteristic (ROC) Curves")
    roc_path = os.path.join(BASE_DIR, "roc_curves.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="Comparative ROC Curves for VGG16, ResNet50, and EfficientNetB0", use_container_width=True)
    else:
        st.warning("ROC curve asset not found. Please ensure `evaluate_models.py` has been executed.")