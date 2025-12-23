# Pet Matting & Background Swap Project

A sophisticated deep learning solution for automatically removing backgrounds from pet images with professional-quality results. This project features a web application supporting transparent cutout generation, model comparison, and interactive background swapping.

---

## 🎯 Team Contributions

This is a **2-member collaborative project** with clearly distinct contributions:

### 👨‍💻 Vikhas: MODNet Implementation
- **Focus**: Real-time, lightweight matting optimized for speed and efficiency
- **Model Architecture**: MODNet (Mobile Optimized Deep Network)
- **Code Location**: `code/modnet/`
- **Model Weights**: `models/modnet/modnet_pet_matting.keras`
- **Benchmark Performance**: IoU 0.7237, Dice 0.8290, MAE 0.0807

### 👨‍💻 Lalitha: FBA Matting Implementation
- **Focus**: High-quality, detail-oriented matting for complex fur and hair edges
- **Model Architecture**: FBA (Foreground-Background-Alpha) Matting with U-Net segmentation
- **Code Location**: `code/fba/`
- **Model Weights**: 
  - FBA Refinement: `models/fba/fba_pet_final.pth`
  - U-Net Segmentation: `models/fba/unet/pet_unet_improved_final.keras`
- **Benchmark Performance**: IoU 0.7500, Dice 0.8462, MAE 0.0869
- **Methodology**:
  - **Pseudo-Label Generation**: Utilized improved U-Net predictions to generate soft alpha mattes for training.
  - **Trimap Generation**: Dynamic trimap generation with random unknown region width to handle varying uncertainty.
  - **FBA Refinement**: Fine-tuned FBA model with ResNet-50 backbone using the generated pseudo-labels.
- **Novelty**:
  - **Laplacian Pyramid Loss**: Introduced a multi-scale loss function to capture high-frequency fur details.
  - **Domain Adaptation**: Fine-tuned a general matting model specifically for pet imagery.
  - **Self-Training**: Leveraged self-generated pseudo-labels to bridge the gap between coarse masks and soft mattes.

---

## ✨ Features

1. **🎨 Transparent Cutout Generator**
   - Upload a pet image and generate a high-quality PNG with transparent background
   - Choose between MODNet or FBA model
   - View alpha matte and processing pipeline visualization
   - Download results instantly

2. **⚖️ Model Comparison**
   - Side-by-side comparison of MODNet and FBA results
   - Visual and quantitative performance analysis
   - Automatic winner determination based on metrics

3. **🖼️ Interactive Background Swap**
   - Drag-and-drop pet positioning
   - Scroll to resize foreground
   - Real-time preview with transparent checkered background
   - Export high-resolution composites

---

## 📄 File Descriptions

### 🌐 Web Application (`webapp/`)

#### `app.py`
**Main Flask server** with three API endpoints:
- `/api/cutout`: Generates transparent cutouts using selected model (MODNet/FBA)
- `/api/compare`: Side-by-side comparison of both models
- `/api/composite`: Creates foreground + background composites
- Handles model loading, image processing, and metric calculation

#### `fba_model.py`
**FBA Pipeline Integration**:
- `SimpleFBAPipeline`: Combines U-Net segmentation with FBA refinement
- `load_fba_pipeline()`: Loads model weights and initializes pipeline
- `predict_fba_pipeline()`: Runs inference on input images
- Handles trimap generation and alpha matte refinement

#### `templates/index.html`
**Main UI Template**:
- Three-tab interface (Cutout, Compare, Composite)
- Model selector radio buttons for MODNet/FBA
- Drag-and-drop file upload zones
- Interactive composite editor with positioning controls
- Real-time metrics display

#### `static/script.js`
**Frontend Logic**:
- File upload and preview handling
- API calls to backend (`generateCutout()`, `compareModels()`, `startCompositeEditor()`)
- Interactive drag-and-drop for composite editor
- Dynamic metric updates and UI state management

#### `static/style.css`
**UI Styling**:
- Modern dark theme with gradient accents
- Custom radio buttons for model selection
- Animated transitions and hover effects
- Responsive design for mobile compatibility

---

### 🧠 MODNet Code (`code/modnet/`)

#### `modnet_architecture.py`
**MODNet Model Definition**:
- `MODNet`: MobileNetV2-based encoder-decoder architecture
- `create_modnet()`: Factory function with customizable parameters
- Optimized for real-time inference

#### `modnet_data_utils.py`
**Data Processing**:
- `prepare_modnet_training_data()`: Prepares Oxford-IIIT Pet dataset
- `create_training_dataset()`: Applies augmentation and batching
- Handles segmentation mask preprocessing

#### `train_modnet.py`
**Main Training Script**:
- Full training pipeline with callbacks and logging
- Custom loss functions (BCE + L1 + Gradient Loss)
- Visualization of training progress
- Saves model checkpoints and final weights

#### `evaluation.py`
**Model Evaluation**:
- IoU, Dice, MAE metric calculations
- Batch evaluation on test datasets
- Performance visualization

#### Other MODNet Files:
- `train_modnet_quick.py`: Fast prototyping training script
- `modnet_inference.py`: Standalone inference utilities
- `pipeline.py`: Complete matting pipeline
- `trimap_generation.py`: Trimap creation from masks
- `generate_cutouts.py`: Batch processing script
- `visualize_matting_process.py`: Pipeline visualization
- `demo_cutouts.py`: Interactive demo

---

### 🔬 FBA Code (`code/fba/`)

#### `simple_pipeline.py`
**Simplified FBA Pipeline**:
- Two-stage matting: U-Net → FBA refinement
- Trimap generation utilities
- Handles PyTorch model loading and inference

#### `FBA_Matting/networks/models.py`
**FBA Model Architecture**:
- `MattingModule`: Main FBA matting network
- `fba_encoder()`: ResNet-based feature encoder
- `fba_decoder()`: Pyramid pooling decoder
- `fba_fusion()`: Foreground-background-alpha fusion

#### Other FBA Files:
- `networks/resnet_GN_WS.py`: ResNet with Group Normalization and Weight Standardization
- `networks/layers_WS.py`: Custom weight-standardized layers
- `dataloader.py`: Data loading utilities for training

---

## 🚀 Setup & Usage

### Prerequisites
```bash
pip install tensorflow torch pillow flask numpy scipy
```

### Running the Application

1. **Navigate to webapp directory**:
   ```bash
   cd webapp
   ```

2. **Start the Flask server**:
   ```bash
   python3 app.py
   ```

3. **Open in browser**:
   ```
   http://localhost:4000
   ```

### Using the Web Interface

#### Transparent Cutout:
1. Select model (MODNet or FBA)
2. Upload a pet image
3. Click "Generate Cutout"
4. Download the result

#### Model Comparison:
1. Upload an image
2. Click "Compare Models"
3. View side-by-side results

#### Background Swap:
1. Select model (MODNet or FBA)
2. Upload pet image and background
3. Click "Open Editor"
4. Drag to position, scroll to resize
5. Download composite

---

## 📊 Model Performance

| Model | IoU | Dice | MAE | Speed |
|-------|-----|------|-----|-------|
| **MODNet** | 0.7237 | 0.8290 | 0.0807 | ⚡ Fast |
| **FBA** | 0.7500 | 0.8462 | 0.0869 | 🎯 Accurate |

*Benchmarks on Oxford-IIIT Pet Test Set*

---

## 🎓 Dataset

Trained on **Oxford-IIIT Pet Dataset**:
- 37 pet categories
- ~7,000 images with segmentation masks
- Diverse backgrounds and poses

---

## 📝 License

This project is for educational purposes.

---

## 👥 Contributors

- **Vikhas**: MODNet implementation and training
- **Lalitha**: FBA integration and pipeline development
