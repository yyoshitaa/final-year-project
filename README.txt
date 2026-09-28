🩺 MAMMOGRAPHY AI DECISION SUPPORT SYSTEM
1. Project Overview & Key Features
An advanced, cross-platform Computer-Aided Diagnosis (CAD) prototype designed to assist clinical workflows in breast cancer screening. The system utilizes deep transfer learning architectures (VGG16, ResNet50, and EfficientNetB0) paired with specialized image preprocessing and class-weighted loss optimization to provide reliable, objective secondary readings.

Multi-Architecture Comparison: Evaluates mammogram scans across three distinct deep learning models simultaneously to ensure diagnostic consensus.

Advanced Preprocessing Pipeline: Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance subtle microcalcifications and low-contrast tissue structures.

Robust Data Splitting: Strict patient-level data partitioning prevents data leakage across training, validation, and test sets.

Interactive Web Dashboard: Built using Streamlit for real-time model inspection, side-by-side comparative inference, and global performance tracking.

2. System Architecture & Tech Stack
Deep Learning Framework: PyTorch (torch, torchvision)

Image Processing: OpenCV (opencv-python), Pillow (PIL)

Web Interface: Streamlit

Metrics & Evaluation: Scikit-Learn, Pandas, NumPy, Matplotlib

3. Installation & Deployment Instructions
A. Deployment on Windows
Open Command Prompt (CMD) or PowerShell and navigate to your project directory:
cd C:\Users\thera\Mammography_Breast_Cancer_Project

Create and activate a Python virtual environment:
python -m venv .venv
.venv\Scripts\activate

Install the required dependencies:
pip install --upgrade pip torch torchvision streamlit opencv-python pillow numpy pandas scikit-learn matplotlib

Run the Model Training Pipeline (Optional / if retraining):
python train_models.py

Generate Evaluation Metrics and ROC Curves:
python evaluate_models.py

Launch the Streamlit Web Application:
streamlit run app.py (Opens automatically at http://localhost:8501)

B. Deployment on macOS
Open Terminal and navigate to your project workspace directory:
cd ~/Path/To/Mammography_Breast_Cancer_Project

Create and activate a Python virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install the required dependencies:
pip install --upgrade pip torch torchvision streamlit opencv-python pillow numpy pandas scikit-learn matplotlib

Run the Model Training Pipeline (Optional / if retraining):
python train_models.py

Generate Evaluation Metrics and ROC Curves:
python evaluate_models.py

Launch the Streamlit Web Application:
streamlit run app.py (Launches at http://localhost:8501)

4. Dashboard Navigation Guide
Tab 1 (Model Inspector): Select an individual model architecture, review its design profile, upload a single mammogram scan, and inspect real-time classification probabilities.

Tab 2 (Side-by-Side Comparison): Upload a target scan to run all three models concurrently and analyze model agreement.

Tab 3 (Global Performance & Metrics): Review the comprehensive validation benchmark table and comparative ROC-AUC curves.

5. Clinical Disclaimer
This software is an academic capstone research prototype designed for computer-aided screening evaluation. It must not be used as a standalone diagnostic tool for primary medical decisions.