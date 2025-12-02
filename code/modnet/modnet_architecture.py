"""
MODNet (Matting Objective Decomposition Network) Architecture
Optimized for pet fur/hair matting with fine detail preservation.

Based on: "Is a Green Screen Really Necessary for Real-Time Portrait Matting?"
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class MODNet(keras.Model):
    """
    MODNet architecture with three branches:
    1. Low-resolution branch: Semantic estimation (coarse segmentation)
    2. High-resolution branch: Detail prediction (fine fur details)
    3. Fusion: Combines semantic and detail for final matte
    """
    
    def __init__(self, input_shape=(128, 128, 3), backbone='mobilenetv2'):
        super(MODNet, self).__init__()
        self.input_shape_val = input_shape
        self.backbone_name = backbone
        
        # Build encoder (MobileNetV2 backbone)
        self.encoder = self._build_encoder()
        
        # Build low-resolution branch (semantic estimation)
        self.lr_branch = self._build_lr_branch()
        
        # Build high-resolution branch (detail prediction)
        self.hr_branch = self._build_hr_branch()
        
        # Fusion layers
        self.fusion_conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.fusion_conv2 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.fusion_output = layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        
    def _build_encoder(self):
        """Build MobileNetV2 encoder with multiple output layers"""
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape_val,
            include_top=False,
            weights='imagenet'
        )
        
        # Extract features at different scales
        # MobileNetV2 layer names for feature extraction
        layer_names = [
            'block_1_expand_relu',   # 64x64, 96 channels
            'block_3_expand_relu',   # 32x32, 144 channels
            'block_6_expand_relu',   # 16x16, 192 channels
            'block_13_expand_relu',  # 8x8, 576 channels
            'out_relu'               # 4x4, 1280 channels
        ]
        
        outputs = [base_model.get_layer(name).output for name in layer_names]
        
        encoder = keras.Model(inputs=base_model.input, outputs=outputs)
        
        # Fine-tune only the last few layers
        for layer in encoder.layers[:-30]:
            layer.trainable = False
            
        return encoder
    
    def _build_lr_branch(self):
        """Low-resolution branch for semantic estimation"""
        return keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.UpSampling2D(2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.UpSampling2D(2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        ], name='lr_branch')
    
    def _build_hr_branch(self):
        """High-resolution branch for detail prediction"""
        return keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.Conv2D(1, 1, activation='sigmoid', padding='same')
        ], name='hr_branch')
    
    def call(self, inputs, training=False):
        """
        Forward pass through MODNet
        
        Args:
            inputs: RGB image tensor (batch, H, W, 3)
            training: Whether in training mode
            
        Returns:
            If training: (semantic_pred, detail_pred, matte_pred)
            If inference: matte_pred only
        """
        # Extract multi-scale features from encoder
        f1, f2, f3, f4, f5 = self.encoder(inputs, training=training)
        
        # Low-resolution branch (semantic estimation)
        # Use deepest features for semantic understanding
        semantic_pred = self.lr_branch(f5, training=training)
        
        # Upsample semantic prediction to match input size
        semantic_upsampled = tf.image.resize(
            semantic_pred, 
            [self.input_shape_val[0], self.input_shape_val[1]],
            method='bilinear'
        )
        
        # High-resolution branch (detail prediction)
        # Use shallow features for fine details
        # Upsample f1 to match input resolution
        f1_upsampled = tf.image.resize(
            f1,
            [self.input_shape_val[0], self.input_shape_val[1]],
            method='bilinear'
        )
        hr_features = layers.concatenate([f1_upsampled, inputs])
        detail_pred = self.hr_branch(hr_features, training=training)
        
        # Fusion: Combine semantic and detail predictions
        fusion_input = layers.concatenate([semantic_upsampled, detail_pred, inputs])
        x = self.fusion_conv1(fusion_input)
        x = self.fusion_conv2(x)
        matte_pred = self.fusion_output(x)
        
        if training:
            # Return all three predictions for multi-loss training
            return semantic_upsampled, detail_pred, matte_pred
        else:
            # Return only final matte for inference
            return matte_pred
    
    def get_config(self):
        """Return config for serialization"""
        config = super(MODNet, self).get_config()
        config.update({
            "input_shape": self.input_shape_val,
            "backbone": self.backbone_name
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        input_shape = config.pop('input_shape', (128, 128, 3))
        backbone = config.pop('backbone', 'mobilenetv2')
        return cls(input_shape=input_shape, backbone=backbone)


def modnet_loss(y_true, y_pred_tuple, alpha_semantic=1.0, alpha_detail=1.0, alpha_matte=1.0):
    """
    Multi-objective loss for MODNet training
    
    Args:
        y_true: Ground truth alpha matte
        y_pred_tuple: (semantic_pred, detail_pred, matte_pred)
        alpha_semantic: Weight for semantic loss
        alpha_detail: Weight for detail loss
        alpha_matte: Weight for matte loss
        
    Returns:
        Combined loss value
    """
    semantic_pred, detail_pred, matte_pred = y_pred_tuple
    
    # 1. Semantic Loss (Binary Cross-Entropy)
    # Encourages coarse segmentation
    semantic_loss = tf.reduce_mean(
        keras.losses.binary_crossentropy(y_true, semantic_pred)
    )
    
    # 2. Detail Loss (L1 + Gradient)
    # Encourages fine detail preservation
    detail_l1 = tf.reduce_mean(tf.abs(y_true - detail_pred))
    
    # Gradient loss for edge preservation
    def gradient_loss(true, pred):
        true_grad_x = true[:, :, 1:, :] - true[:, :, :-1, :]
        true_grad_y = true[:, 1:, :, :] - true[:, :-1, :, :]
        pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_grad_y = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        
        grad_loss = tf.reduce_mean(tf.abs(true_grad_x - pred_grad_x)) + \
                   tf.reduce_mean(tf.abs(true_grad_y - pred_grad_y))
        return grad_loss
    
    detail_grad = gradient_loss(y_true, detail_pred)
    detail_loss = detail_l1 + 0.5 * detail_grad
    
    # 3. Matte Loss (L1 + Gradient + Laplacian)
    # Encourages accurate final prediction
    matte_l1 = tf.reduce_mean(tf.abs(y_true - matte_pred))
    matte_grad = gradient_loss(y_true, matte_pred)
    
    # Laplacian loss for smoothness
    def laplacian_loss(true, pred):
        kernel = tf.constant([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        
        true_lap = tf.nn.conv2d(true, kernel, strides=1, padding='SAME')
        pred_lap = tf.nn.conv2d(pred, kernel, strides=1, padding='SAME')
        
        return tf.reduce_mean(tf.abs(true_lap - pred_lap))
    
    matte_lap = laplacian_loss(y_true, matte_pred)
    matte_loss = matte_l1 + 0.5 * matte_grad + 0.25 * matte_lap
    
    # Combined loss
    total_loss = (alpha_semantic * semantic_loss + 
                  alpha_detail * detail_loss + 
                  alpha_matte * matte_loss)
    
    return total_loss


class MODNetTrainer(keras.Model):
    """
    Wrapper model for training MODNet with custom loss
    """
    
    def __init__(self, modnet_model, **kwargs):
        super(MODNetTrainer, self).__init__(**kwargs)
        self.modnet = modnet_model
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_tracker = keras.metrics.MeanAbsoluteError(name="mae")
        
    def call(self, inputs, training=False):
        return self.modnet(inputs, training=training)
    
    def train_step(self, data):
        x, y_true = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred_tuple = self.modnet(x, training=True)
            
            # Calculate loss
            loss = modnet_loss(y_true, y_pred_tuple)
        
        # Backward pass
        gradients = tape.gradient(loss, self.modnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.modnet.trainable_variables))
        
        # Update metrics
        _, _, matte_pred = y_pred_tuple
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, matte_pred)
        
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result()
        }
    
    def test_step(self, data):
        x, y_true = data
        
        # Forward pass
        y_pred_tuple = self.modnet(x, training=True)
        
        # Calculate loss
        loss = modnet_loss(y_true, y_pred_tuple)
        
        # Update metrics
        _, _, matte_pred = y_pred_tuple
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, matte_pred)
        
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result()
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_tracker]


def create_modnet(input_shape=(128, 128, 3), learning_rate=1e-4):
    """
    Create and compile MODNet model for training
    
    Args:
        input_shape: Input image shape
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled MODNetTrainer model
    """
    modnet = MODNet(input_shape=input_shape)
    trainer = MODNetTrainer(modnet)
    
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    
    return trainer


if __name__ == "__main__":
    print("=" * 60)
    print("MODNet Architecture Test")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating MODNet model...")
    model = create_modnet(input_shape=(128, 128, 3))
    
    # Test with dummy data
    print("\n2. Testing forward pass...")
    dummy_input = tf.random.normal((2, 128, 128, 3))
    
    # Training mode (returns 3 outputs)
    semantic, detail, matte = model.modnet(dummy_input, training=True)
    print(f"   Semantic output shape: {semantic.shape}")
    print(f"   Detail output shape: {detail.shape}")
    print(f"   Matte output shape: {matte.shape}")
    
    # Inference mode (returns 1 output)
    matte_only = model.modnet(dummy_input, training=False)
    print(f"   Inference output shape: {matte_only.shape}")
    
    # Test loss calculation
    print("\n3. Testing loss calculation...")
    dummy_gt = tf.random.uniform((2, 128, 128, 1), 0, 1)
    loss = modnet_loss(dummy_gt, (semantic, detail, matte))
    print(f"   Loss value: {loss.numpy():.4f}")
    
    # Model summary
    print("\n4. Model summary:")
    print(f"   Total parameters: {model.modnet.count_params():,}")
    
    print("\n" + "=" * 60)
    print("MODNet architecture test completed successfully!")
    print("=" * 60)
