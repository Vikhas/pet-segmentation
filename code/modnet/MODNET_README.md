# MODNet Fine-Tuning for Pet Fur Matting

This directory contains a complete implementation of **MODNet (Matting Objective Decomposition Network)** fine-tuned specifically for high-quality pet fur/hair matting to achieve clear, transparent cutouts.

## 🎯 What is MODNet?

MODNet is a state-of-the-art portrait matting model that decomposes the matting task into:
1. **Semantic Estimation**: Coarse segmentation at low resolution
2. **Detail Prediction**: Fine detail capture at high resolution  
3. **Fusion**: Combines both for accurate alpha mattes

This makes it particularly effective for capturing fine details like pet fur and whiskers.

## 📁 New Files

### Core Implementation

- **`modnet_architecture.py`**: Full MODNet implementation
  - MobileNetV2 backbone (pre-trained on ImageNet)
  - Multi-branch architecture (semantic + detail + fusion)
  - Custom multi-objective loss (semantic, detail, matte)
  - ~5.7M parameters

- **`modnet_data_utils.py`**: Data preparation utilities
  - Smart trimap generation
  - Fur-specific data augmentation
  - Optimized TensorFlow dataset pipeline

- **`train_modnet.py`**: Comprehensive training script
  - 40 epochs with early stopping
  - Real-time visualization callbacks
  - Automatic checkpointing
  - Detailed logging

- **`modnet_inference.py`**: Production inference script
  - Single image processing
  - Batch processing
  - Automatic resizing and postprocessing

- **`compare_models.py`**: Model comparison tool
  - Side-by-side visual comparison
  - Quantitative metrics (MAE, IoU, Dice)
  - Difference maps

## 🚀 Quick Start

### 1. Test the Architecture

```bash
cd code
python3 modnet_architecture.py
```

Expected output: ✅ All tests pass, model has 5,680,995 parameters

### 2. Train MODNet (2-4 hours)

```bash
python3 train_modnet.py
```

Training will:
- Use 2000 training samples, 200 validation samples
- Apply data augmentation (flips, color jitter, etc.)
- Save best model to `../models/modnet_pet_matting.keras`
- Generate visualizations in `../outputs/modnet_training_vis/`
- Log metrics to `../outputs/modnet_logs/`

### 3. Compare Models

After training completes:

```bash
python3 compare_models.py
```

This generates side-by-side comparisons of:
- Simple matting model (current)
- MODNet (new)

### 4. Generate Cutouts

```python
from modnet_inference import load_modnet, visualize_inference

model = load_modnet('../models/modnet_pet_matting.keras')
visualize_inference('path/to/pet.jpg', model, 'output.png')
```

## 📊 Expected Performance

Based on MODNet architecture and training setup:

| Metric | Simple Model | MODNet (Expected) |
|--------|--------------|-------------------|
| MAE    | 0.1164       | **< 0.08**       |
| IoU    | 0.8369       | **> 0.88**       |
| Dice   | 0.9068       | **> 0.93**       |

**Key Improvements:**
- ✅ Better fur detail preservation
- ✅ Smoother edges
- ✅ More accurate alpha gradients
- ✅ Handles various fur types better

## 🔧 Architecture Details

### Encoder (MobileNetV2)
- Pre-trained on ImageNet
- Extracts multi-scale features
- Last 30 layers fine-tunable

### Low-Resolution Branch
- Processes 4x4 features
- Semantic segmentation
- Upsampled to full resolution

### High-Resolution Branch  
- Processes 64x64 features
- Detail prediction
- Preserves fine fur details

### Fusion Module
- Combines semantic + detail
- 3-layer refinement
- Final alpha matte output

### Loss Function
```
Total Loss = α₁·Semantic Loss + α₂·Detail Loss + α₃·Matte Loss

where:
- Semantic Loss = Binary Cross-Entropy
- Detail Loss = L1 + Gradient Loss
- Matte Loss = L1 + Gradient + Laplacian Loss
```

## 📈 Training Monitoring

During training, check:

1. **Visualizations**: `../outputs/modnet_training_vis/`
   - Generated every 2 epochs
   - Shows predictions vs ground truth

2. **Logs**: `../outputs/modnet_logs/training_log.txt`
   - Epoch-by-epoch metrics
   - Loss and MAE tracking

3. **TensorBoard**: 
   ```bash
   tensorboard --logdir ../outputs/modnet_logs
   ```

## 🎨 Data Augmentation

The training applies:
- Random horizontal flips
- Random brightness (±20%)
- Random contrast (80-120%)
- Random saturation (80-120%)
- Random hue (±10%)

This helps the model generalize to various lighting conditions and pet appearances.

## 💾 Model Outputs

After training:
- `../models/modnet_pet_matting.keras` - Final trained model
- `../models/modnet_checkpoints/modnet_best.keras` - Best checkpoint
- `../outputs/modnet_training_vis/` - Training visualizations
- `../outputs/modnet_logs/` - Training logs

## 🔍 Troubleshooting

**Out of Memory Error:**
```python
# In train_modnet.py, reduce:
BATCH_SIZE = 4  # from 8
NUM_TRAIN_SAMPLES = 1000  # from 2000
```

**Training Too Slow:**
```python
# Reduce epochs:
EPOCHS = 20  # from 40
```

**Model Not Learning:**
- Check `training_log.txt` for loss trends
- View visualizations to see if predictions improve
- Ensure mask mean is > 0 (not all zeros)

## 📚 References

- [MODNet Paper](https://arxiv.org/abs/2011.11961)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## ✨ Next Steps

After training completes:
1. Run `compare_models.py` to evaluate improvements
2. Use `modnet_inference.py` for production cutouts
3. Integrate into your existing pipeline via `pipeline.py`
