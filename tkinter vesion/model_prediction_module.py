"""
===============================================================================
MODEL PREDICTION MODULE - Machine Learning Predictions
===============================================================================
Handles all machine learning model operations including:
- Model loading with custom objects (dice_loss, dice_coef)
- Sliding window prediction for large images
- Probability map generation
- GeoTIFF output with geospatial metadata preservation
- Batch processing
===============================================================================
"""

import numpy as np
import os
import rasterio
from datetime import datetime
import time

# TensorFlow/Keras imports
import tensorflow as tf
import keras.backend as K
from tensorflow import keras


class ModelPredictor:
    """Handles machine learning model predictions"""
    
    def __init__(self):
        """Initialize the model predictor"""
        self.custom_objects = self._create_custom_objects()
        
    def _create_custom_objects(self):
        """Create custom objects dictionary for model loading"""
        
        def dice_loss(y_true, y_pred, smooth=1e-6):
            """Dice Loss for Segmentation"""
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        def dice_coef(y_true, y_pred, smooth=1e-6):
            """Dice Coefficient Metric"""
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        def iou_score(y_true, y_pred, smooth=1e-6):
            """IoU Score"""
            intersection = K.sum(K.abs(y_true * y_pred))
            union = K.sum(y_true) + K.sum(y_pred) - intersection
            return (intersection + smooth) / (union + smooth)
        
        return {
            'dice_loss': dice_loss,
            'dice_coef': dice_coef,
            'iou_score': iou_score
        }
    
    def load_model(self, model_path):
        """
        Load a Keras model with custom objects.
        
        Args:
            model_path: Path to .keras or .h5 model file
            
        Returns:
            tuple: (success, message, model, input_size)
        """
        try:
            print(f"\n{'='*80}")
            print(f"Loading model: {os.path.basename(model_path)}")
            print(f"{'='*80}")
            
            # Load model with custom objects
            model = keras.models.load_model(
                model_path,
                custom_objects=self.custom_objects
            )
            
            # Detect input size
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            input_size = input_shape[1]  # Assumes square patches
            
            print(f"✅ Model loaded successfully!")
            print(f"   Input shape: {input_shape}")
            print(f"   Patch size: {input_size}x{input_size}")
            print(f"{'='*80}\n")
            
            return True, f"✅ Model loaded: {input_size}x{input_size} patches", model, input_size
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            print(error_msg)
            return False, error_msg, None, None
    
    def get_model_details(self, model):
        """
        Get detailed information about the model.
        
        Args:
            model: Keras model
            
        Returns:
            str: Formatted model details
        """
        try:
            # Capture model summary
            import io
            buffer = io.StringIO()
            model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            summary = buffer.getvalue()
            
            # Get input/output shapes
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            # Count parameters
            total_params = model.count_params()
            
            details = f"""
MODEL DETAILS
{'='*60}

Input Shape:  {input_shape}
Output Shape: {output_shape}
Total Parameters: {total_params:,}

ARCHITECTURE
{'='*60}
{summary}
"""
            return details
            
        except Exception as e:
            return f"Error getting model details: {e}"
    
    def preprocess_for_model(self, composite_image):
        """
        Preprocess image for model (NO preprocessing - model handles it).
        
        Args:
            composite_image: Raw composite image array
            
        Returns:
            tuple: (preprocessed_image, log_message)
        """
        print("\n" + "="*80)
        print("🎯 NO PREPROCESSING - USING RAW DATA!")
        print("="*80)
        print("✅ Model handles all preprocessing internally")
        print("="*80 + "\n")
        
        # Just convert to float32 and handle NaN/Inf
        raw_data = composite_image.astype(np.float32)
        
        # Only replace NaN/Inf with 0
        raw_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
        
        # Print stats
        print(f"📊 Raw Data Statistics:")
        print(f"   Shape: {raw_data.shape}")
        print(f"   Dtype: {raw_data.dtype}")
        print(f"   Range: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
        print(f"   Mean: {raw_data.mean():.4f}")
        print(f"   Std: {raw_data.std():.4f}")
        
        log_message = (
            "\n[NO PREPROCESSING APPLIED]\n"
            "✅ Using RAW data - Model has built-in preprocessing layers\n"
            f"   Shape: {raw_data.shape}\n"
            f"   Range: [{raw_data.min():.4f}, {raw_data.max():.4f}]\n"
        )
        
        return raw_data, log_message
    
    def sliding_window_predict(self, model, image, window_size=128, stride=64):
        """
        Perform sliding window prediction on large image.
        
        Args:
            model: Keras model
            image: Input image (H, W, C)
            window_size: Size of sliding window
            stride: Stride for sliding window
            
        Returns:
            np.array: Probability map (H, W)
        """
        try:
            print(f"\n🔍 Sliding Window Prediction")
            print(f"   Image size: {image.shape}")
            print(f"   Window: {window_size}x{window_size}")
            print(f"   Stride: {stride}")
            
            h, w = image.shape[:2]
            
            # Initialize probability and count maps
            prob_map = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.float32)
            
            # Calculate number of windows
            n_rows = ((h - window_size) // stride) + 1
            n_cols = ((w - window_size) // stride) + 1
            total_windows = n_rows * n_cols
            
            print(f"   Total windows: {total_windows}")
            
            # Sliding window
            window_count = 0
            for i in range(0, h - window_size + 1, stride):
                for j in range(0, w - window_size + 1, stride):
                    # Extract window
                    window = image[i:i+window_size, j:j+window_size]
                    
                    # Preprocess
                    window_processed, _ = self.preprocess_for_model(window)
                    
                    # Add batch dimension
                    window_batch = np.expand_dims(window_processed, axis=0)
                    
                    # Predict
                    pred = model.predict(window_batch, verbose=0)
                    
                    # Get probability (squeeze to 2D)
                    pred_2d = np.squeeze(pred)
                    
                    # Add to probability map
                    prob_map[i:i+window_size, j:j+window_size] += pred_2d
                    count_map[i:i+window_size, j:j+window_size] += 1
                    
                    window_count += 1
                    if window_count % 100 == 0:
                        print(f"   Processed {window_count}/{total_windows} windows")
            
            # Average overlapping predictions
            prob_map = np.divide(
                prob_map,
                count_map,
                out=np.zeros_like(prob_map),
                where=count_map != 0
            )
            
            # Handle NaN/Inf
            prob_map = np.where(np.isfinite(prob_map), prob_map, 0.0)
            
            # Clip to [0, 1]
            prob_map = np.clip(prob_map, 0.0, 1.0)
            
            print(f"✅ Prediction complete")
            print(f"   Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
            
            return prob_map
            
        except Exception as e:
            print(f"❌ Error in sliding window prediction: {e}")
            raise
    
    def save_geotiff_probability(self, pred_mask, original_img_path, output_path):
        """
        Save probability map as GeoTIFF with geospatial metadata.
        
        Args:
            pred_mask: Probability map (H, W) or (H, W, 1)
            original_img_path: Path to original image (for metadata)
            output_path: Output file path
            
        Returns:
            str: Path to saved file
        """
        try:
            # Open original to get metadata
            with rasterio.open(original_img_path) as src:
                profile = src.profile.copy()
                transform = src.transform
                crs = src.crs
            
            # Update profile
            profile.update({
                'count': 1,
                'dtype': 'float32',
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'nodata': None
            })
            
            # Ensure 2D
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[:, :, 0]
            
            # Clean data
            pred_mask_clean = pred_mask.copy()
            pred_mask_clean[~np.isfinite(pred_mask_clean)] = 0.0
            pred_mask_clean = np.clip(pred_mask_clean, 0.0, 1.0)
            
            # Save
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(pred_mask_clean.astype(np.float32), 1)
                dst.set_band_description(1, 'Snow Probability (0.0-1.0)')
                dst.update_tags(
                    prediction_type='snow_detection',
                    data_type='probability_float32',
                    value_range='0.0_to_1.0'
                )
            
            print(f"✅ Saved probability map: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving GeoTIFF: {e}")
            raise
    
    def predict_single_image(self, model, img_path, output_folder, window_size=128, stride=64):
        """
        Run prediction on a single image.
        
        Args:
            model: Keras model
            img_path: Input image path
            output_folder: Output folder path
            window_size: Window size for sliding window
            stride: Stride for sliding window
            
        Returns:
            tuple: (success, message, output_path)
        """
        try:
            print(f"\n{'='*80}")
            print(f"Predicting: {os.path.basename(img_path)}")
            print(f"{'='*80}")
            
            start_time = time.time()
            
            # Read image
            with rasterio.open(img_path) as src:
                image = src.read()
                
                # Transpose to (H, W, C)
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                
                image = image.astype(np.float32)
                image = np.where(np.isfinite(image), image, 0.0)
            
            # Run sliding window prediction
            prob_map = self.sliding_window_predict(model, image, window_size, stride)
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_filename = f"{base_name}_prediction_probability.tif"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save result
            self.save_geotiff_probability(prob_map, img_path, output_path)
            
            # Calculate statistics
            elapsed = time.time() - start_time
            mean_prob = np.mean(prob_map)
            high_confidence = np.sum(prob_map > 0.5)
            total_pixels = prob_map.size
            percentage = (high_confidence / total_pixels * 100) if total_pixels > 0 else 0
            
            print(f"\n📊 Prediction Statistics:")
            print(f"   Mean probability: {mean_prob:.4f}")
            print(f"   High confidence pixels (>0.5): {high_confidence} ({percentage:.2f}%)")
            print(f"   Processing time: {elapsed:.2f}s")
            print(f"{'='*80}\n")
            
            return True, f"✅ Prediction saved to {output_filename}", output_path
            
        except Exception as e:
            error_msg = f"❌ Error predicting image: {str(e)}"
            print(error_msg)
            return False, error_msg, None
    
    def predict_batch(self, model, input_folder, output_folder, image_list, window_size=128, stride=64):
        """
        Run predictions on multiple images.
        
        Args:
            model: Keras model
            input_folder: Input folder path
            output_folder: Output folder path
            image_list: List of image filenames to process
            window_size: Window size for sliding window
            stride: Stride for sliding window
            
        Returns:
            tuple: (success, message)
        """
        try:
            print(f"\n{'='*80}")
            print(f"BATCH PREDICTION")
            print(f"{'='*80}")
            print(f"Images to process: {len(image_list)}")
            print(f"Output folder: {output_folder}")
            print(f"{'='*80}\n")
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each image
            success_count = 0
            start_time = time.time()
            
            for i, filename in enumerate(image_list, 1):
                print(f"\n[{i}/{len(image_list)}] Processing: {filename}")
                
                img_path = os.path.join(input_folder, filename)
                
                success, msg, output_path = self.predict_single_image(
                    model,
                    img_path,
                    output_folder,
                    window_size,
                    stride
                )
                
                if success:
                    success_count += 1
            
            # Summary
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"BATCH PREDICTION COMPLETE")
            print(f"{'='*80}")
            print(f"Successful: {success_count}/{len(image_list)}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(image_list):.2f}s")
            print(f"{'='*80}\n")
            
            if success_count == len(image_list):
                return True, f"✅ Successfully processed all {len(image_list)} images"
            else:
                return False, f"⚠️ Processed {success_count}/{len(image_list)} images"
                
        except Exception as e:
            return False, f"❌ Error in batch prediction: {str(e)}"


# ============================================================
# MAIN ENTRY POINT (for testing)
# ============================================================
if __name__ == "__main__":
    # Test the model predictor
    predictor = ModelPredictor()
    print("Model Prediction Module loaded successfully!")
    print("\nSupported operations:")
    print("  - Model loading (.keras, .h5)")
    print("  - Sliding window prediction")
    print("  - Probability map generation")
    print("  - Batch processing")
    print("  - GeoTIFF output with metadata")
