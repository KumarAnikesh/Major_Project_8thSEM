"""
SAFE SOLUTION - No Crashes!
============================
Removes all repaint() calls that cause segmentation faults
Uses ONLY setValue() and lets Qt handle repainting naturally
"""

import numpy as np
import os
import tensorflow as tf
import time

from utils_core_FIXED import (
    preprocess_for_model,
    load_composite_image_8_bands,
    check_composite_exists,
    save_geotiff_probability_only,
    CUSTOM_OBJECTS
)

def load_keras_model(model_path, log_callback=None):
    """Load Keras model"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    try:
        log(f"📦 Loading model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        log(f"✅ Model loaded successfully!")
        log(f"   Input shape: {model.input_shape}")
        log(f"   Output shape: {model.output_shape}")
        return model
    except Exception as e:
        log(f"❌ Error loading model: {e}")
        return None

def predict_with_sliding_window(model, img_data, window_size, stride, target_channels, 
                                progress_callback=None, log_callback=None):
    """
    SAFE VERSION: Update every 10 windows, no fancy tricks
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    height, width = img_data.shape[0], img_data.shape[1]
    
    # Prepare channels
    if img_data.shape[2] != target_channels:
        if img_data.shape[2] > target_channels:
            img_data = img_data[:, :, :target_channels]
        else:
            padding_needed = target_channels - img_data.shape[2]
            padding = np.repeat(img_data[:, :, -1:], padding_needed, axis=2)
            img_data = np.concatenate([img_data, padding], axis=2)
    
    # Initialize maps
    dummy_input = np.zeros((1, window_size, window_size, target_channels), dtype=np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    
    if len(dummy_output.shape) == 4:
        output_channels = dummy_output.shape[-1]
        prediction_map = np.zeros((height, width, output_channels), dtype=np.float32)
    else:
        prediction_map = np.zeros((height, width), dtype=np.float32)
    
    count_map = np.zeros((height, width), dtype=np.float32)
    
    total_windows = ((height - window_size) // stride + 1) * ((width - window_size) // stride + 1)
    current_window = 0
    
    log(f"🔬 Processing {total_windows} windows (SAFE MODE - no crashes)...")
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = img_data[y:y+window_size, x:x+window_size, :]
            
            if window.shape[0] != window_size or window.shape[1] != window_size:
                continue
            
            window_batch = np.expand_dims(window, axis=0)
            prediction = model.predict(window_batch, verbose=0)
            
            if len(prediction.shape) == 4:
                pred_window = prediction[0]
                prediction_map[y:y+window_size, x:x+window_size, :] += pred_window
            else:
                pred_value = prediction[0]
                if isinstance(pred_value, (list, np.ndarray)):
                    pred_value = pred_value[0] if len(pred_value) > 0 else 0
                prediction_map[y:y+window_size, x:x+window_size] += pred_value
            
            count_map[y:y+window_size, x:x+window_size] += 1
            current_window += 1
            
            # ✅ SAFE: Update every 10 windows only
            if progress_callback and current_window % 10 == 0:
                current_pct = int((current_window * 100) / total_windows)
                progress_callback(current_window, total_windows, current_pct)
    
    # Final update
    if progress_callback:
        progress_callback(total_windows, total_windows, 100)
    
    # Average overlaps
    count_map[count_map == 0] = 1
    if len(prediction_map.shape) == 3:
        for c in range(prediction_map.shape[2]):
            prediction_map[:, :, c] /= count_map
    else:
        prediction_map /= count_map
    
    log(f"✅ Complete!")
    
    return prediction_map, count_map

def predict_single_image(model, image_path, window_size, stride, output_folder=None, 
                        progress_callback=None, log_callback=None):
    """Run prediction - SAFE VERSION"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    try:
        log("="*80)
        log("🚀 STARTING PREDICTION - SAFE MODE")
        log("="*80)
        
        start_time = time.time()
        
        is_composite = '_composite' in os.path.basename(image_path).lower()
        
        if is_composite:
            composite_path = image_path
            log(f"\n📁 Using composite: {os.path.basename(composite_path)}")
            
            import rasterio
            with rasterio.open(composite_path) as src:
                if src.count != 8:
                    return {
                        'success': False,
                        'error': f'Expected 8-band composite, got {src.count} bands'
                    }
                composite = src.read()
                composite = np.transpose(composite, (1, 2, 0))
                log(f"✅ Loaded 8-band composite: {composite.shape}")
            
            reference_path = composite_path
        else:
            composite = load_composite_image_8_bands(image_path)
            reference_path = image_path
            
            if composite is None:
                return {
                    'success': False,
                    'error': 'Composite not found. Please run "All Indices" first.'
                }
        
        height, width, channels = composite.shape
        log(f"\n📊 Image: {height} x {width} | Channels: {channels}")
        
        if channels != 8:
            return {
                'success': False,
                'error': f'Expected 8 channels, got {channels}'
            }
        
        preprocessed, _ = preprocess_for_model(composite)
        target_channels = model.input_shape[-1]
        
        log(f"\n🔬 Running prediction (Window: {window_size}x{window_size}, Stride: {stride})")
        
        pred_map, count_map = predict_with_sliding_window(
            model, preprocessed, window_size, stride, target_channels,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        log(f"\n📊 Prediction statistics:")
        log(f"   Shape: {pred_map.shape}")
        log(f"   Mean confidence: {pred_map.mean():.3f}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        base_name = base_name.replace('_processed_composite', '').replace('_composite', '')
        output_path = os.path.join(output_folder, f"{base_name}_prediction_probability.tif")
        
        log(f"💾 Saving: {os.path.basename(output_path)}")
        save_geotiff_probability_only(pred_map, reference_path, output_path)
        
        elapsed_time = time.time() - start_time
        log(f"✅ Complete in {elapsed_time:.1f}s")
        log("="*80)
        
        return {
            'success': True,
            'output_path': output_path,
            'prediction_map': pred_map,
            'processing_time': elapsed_time,
            'mean_confidence': float(pred_map.mean())
        }
        
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def predict_batch_images(model, folder_path, window_size, stride, output_folder, 
                        progress_callback=None, log_callback=None):
    """Batch prediction - SAFE VERSION"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    try:
        log("="*80)
        log("📦 STARTING BATCH PREDICTION - SAFE MODE")
        log("="*80)
        
        all_files = os.listdir(folder_path)
        tiff_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
        
        log(f"\n🔍 Scanning folder: {folder_path}")
        log(f"   Total TIFF files found: {len(tiff_files)}")
        
        image_files = []
        
        for f in tiff_files:
            if '_composite' not in f.lower() and '_ndsi' not in f.lower() and \
               '_ndvi' not in f.lower() and '_ndwi' not in f.lower() and \
               '_prediction' not in f.lower():
                filepath = os.path.join(folder_path, f)
                if check_composite_exists(filepath):
                    image_files.append(f)
                    log(f"   ✓ Found original with composite: {f}")
        
        if not image_files:
            log("\n   No original images found. Looking for composite files...")
            for f in tiff_files:
                if '_processed_composite' in f.lower() or \
                   (f.lower().endswith('_composite.tif') or f.lower().endswith('_composite.tiff')):
                    filepath = os.path.join(folder_path, f)
                    try:
                        import rasterio
                        with rasterio.open(filepath) as src:
                            if src.count == 8:
                                image_files.append(f)
                                log(f"   ✓ Found 8-band composite: {f}")
                    except Exception as e:
                        log(f"   ✗ Skipped: {f} - {str(e)}")
        
        if not image_files:
            return {
                'success': False,
                'error': 'No valid images found.'
            }
        
        log(f"\n✅ Found {len(image_files)} valid images to process")
        
        os.makedirs(output_folder, exist_ok=True)
        
        results = []
        success_count = 0
        start_time = time.time()
        
        for idx, filename in enumerate(image_files):
            if progress_callback:
                progress_callback(idx + 1, len(image_files), filename)
            
            image_path = os.path.join(folder_path, filename)
            result = predict_single_image(
                model, image_path, window_size, stride, output_folder,
                progress_callback=None,
                log_callback=log_callback
            )
            
            if result['success']:
                success_count += 1
                log(f"   ✅ Success: {filename}")
            else:
                log(f"   ❌ Failed: {filename}")
            
            results.append({
                'filename': filename,
                'success': result['success'],
                'output_path': result.get('output_path'),
                'error': result.get('error')
            })
        
        elapsed_time = time.time() - start_time
        
        log("\n" + "="*80)
        log("📊 BATCH PREDICTION COMPLETE")
        log("="*80)
        log(f"   Total images: {len(image_files)}")
        log(f"   Successful: {success_count}")
        log(f"   Time: {elapsed_time:.2f} seconds")
        log("="*80)
        
        return {
            'success': True,
            'total': len(image_files),
            'successful': success_count,
            'failed': len(image_files) - success_count,
            'results': results,
            'processing_time': elapsed_time
        }
        
    except Exception as e:
        log(f"❌ Batch prediction failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_model_info(model):
    """Get model information"""
    try:
        info = {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'patch_size': model.input_shape[1],
            'input_channels': model.input_shape[-1],
            'output_channels': model.output_shape[-1] if len(model.output_shape) > 2 else 1,
            'total_params': model.count_params()
        }
        return info
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def detect_optimal_stride(window_size):
    """Calculate optimal stride"""
    return max(1, window_size // 2)