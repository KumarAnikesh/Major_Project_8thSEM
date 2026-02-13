"""
Core Utility Functions - Framework Independent
===============================================
These functions work with both Tkinter and PyQt6
All geospatial, mathematical, and data processing functions

FIXED VERSION - All functions from original code included
"""

import numpy as np
import rasterio
import os
from datetime import datetime
import tensorflow as tf
import keras.backend as K

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

# Scaling factor for Sentinel-2 bands (divide by this to get 0-1 range)
SCALING_FACTOR = 10000.0

# ============================================================
# CUSTOM OBJECTS FOR MODEL LOADING
# ============================================================

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

CUSTOM_OBJECTS = {
    'dice_loss': dice_loss,
    'dice_coef': dice_coef
}

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_for_model(composite_image):
    """
    No preprocessing - model handles internally
    Returns: (raw_image, log_message)
    """
    print("\n" + "="*80)
    print("🎯 NO PREPROCESSING - USING RAW DATA!")
    print("="*80)
    
    raw_data = composite_image.astype(np.float32)
    raw_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
    
    print(f"📊 Raw Data Statistics:")
    print(f"   Shape: {raw_data.shape}")
    print(f"   Range: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"   Mean: {raw_data.mean():.4f}")
    print(f"   Std: {raw_data.std():.4f}")
    
    log_message = (
        f"\n[NO PREPROCESSING APPLIED]\n"
        f"✅ Using RAW data\n"
        f"   Shape: {raw_data.shape}\n"
        f"   Range: [{raw_data.min():.4f}, {raw_data.max():.4f}]\n"
        f"   Mean: {raw_data.mean():.4f}\n"
    )
    
    return raw_data, log_message

# ============================================================
# BAND DETECTION FUNCTIONS
# ============================================================

def detect_band_by_type(src, band_type):
    """
    Detect band by its type using metadata/descriptions
    
    Args:
        src: rasterio dataset
        band_type: 'GREEN', 'RED', 'NIR', 'SWIR', 'BLUE'
    
    Returns:
        band_index (1-based) or None
    """
    if src is None or src.closed:
        return None
    
    try:
        band_descriptions = src.descriptions
        band_count = src.count
    except Exception:
        return None

    patterns = {
        'GREEN': ['B3', 'GREEN', 'GRN', 'BAND3', 'BAND 3', 'SR_B3'],
        'RED':   ['B4', 'RED', 'BAND4', 'BAND 4', 'SR_B4'],
        'NIR':   ['B8', 'NIR', 'BAND8', 'BAND 8', 'B8A', 'SR_B8', 'NEAR_INFRARED'],
        'SWIR':  ['B11', 'SWIR', 'SWIR1', 'BAND11', 'BAND 11', 'SR_B11'],
        'BLUE':  ['B2', 'BLUE', 'BLU', 'BAND2', 'BAND 2', 'SR_B2']
    }

    if band_type not in patterns:
        return None

    if band_descriptions:
        for i, desc in enumerate(band_descriptions):
            if desc:
                desc_upper = str(desc).upper().strip()
                
                for pattern in patterns[band_type]:
                    if desc_upper == pattern.upper():
                        return i + 1
                
                for pattern in patterns[band_type]:
                    if pattern.upper() in desc_upper:
                        return i + 1

    return None

# Band cache for performance
_band_cache = {}

def cached_detect_band(src, band_type):
    """Cached band detection"""
    global _band_cache
    
    if src is None or src.closed:
        return None
    
    try:
        key = src.name
        if key not in _band_cache:
            _band_cache[key] = {}
        cache = _band_cache[key]
        if band_type not in cache:
            cache[band_type] = detect_band_by_type(src, band_type)
        return cache[band_type]
    except Exception:
        return detect_band_by_type(src, band_type)

def get_required_bands(src, index_type):
    """
    Get required bands for index calculation
    
    Args:
        src: rasterio dataset
        index_type: 'NDSI', 'NDVI', 'NDWI'
    
    Returns:
        dict with band information
    """
    result = {
        'success': False,
        'bands': {},
        'info': ''
    }
    
    if index_type == 'NDSI':
        green_idx = cached_detect_band(src, 'GREEN')
        swir_idx = cached_detect_band(src, 'SWIR')
        
        if not green_idx or not swir_idx:
            missing = []
            if not green_idx:
                missing.append('GREEN (B3)')
            if not swir_idx:
                missing.append('SWIR (B11)')
            result['info'] = f"Missing bands: {', '.join(missing)}"
            return result
        
        result['bands'] = {
            'GREEN': {'index': green_idx, 'data': src.read(green_idx)},
            'SWIR': {'index': swir_idx, 'data': src.read(swir_idx)}
        }
        result['success'] = True
        
    elif index_type == 'NDVI':
        red_idx = cached_detect_band(src, 'RED')
        nir_idx = cached_detect_band(src, 'NIR')
        
        if not red_idx or not nir_idx:
            missing = []
            if not red_idx:
                missing.append('RED (B4)')
            if not nir_idx:
                missing.append('NIR (B8)')
            result['info'] = f"Missing bands: {', '.join(missing)}"
            return result
        
        result['bands'] = {
            'RED': {'index': red_idx, 'data': src.read(red_idx)},
            'NIR': {'index': nir_idx, 'data': src.read(nir_idx)}
        }
        result['success'] = True
        
    elif index_type == 'NDWI':
        green_idx = cached_detect_band(src, 'GREEN')
        nir_idx = cached_detect_band(src, 'NIR')
        
        if not green_idx or not nir_idx:
            missing = []
            if not green_idx:
                missing.append('GREEN (B3)')
            if not nir_idx:
                missing.append('NIR (B8)')
            result['info'] = f"Missing bands: {', '.join(missing)}"
            return result
        
        result['bands'] = {
            'GREEN': {'index': green_idx, 'data': src.read(green_idx)},
            'NIR': {'index': nir_idx, 'data': src.read(nir_idx)}
        }
        result['success'] = True
    
    return result

# ============================================================
# INDEX CALCULATION FUNCTIONS
# ============================================================

def calculate_ndsi(green_band, swir_band):
    """Calculate NDSI (Normalized Difference Snow Index)"""
    green = green_band.astype(float)
    swir = swir_band.astype(float)
    
    ndsi = np.where((green + swir) != 0, (green - swir) / (green + swir), 0)
    return ndsi

def calculate_ndvi(red_band, nir_band):
    """Calculate NDVI (Normalized Difference Vegetation Index)"""
    red = red_band.astype(float)
    nir = nir_band.astype(float)
    
    ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
    return ndvi

def calculate_ndwi(green_band, nir_band):
    """Calculate NDWI (Normalized Difference Water Index)"""
    green = green_band.astype(float)
    nir = nir_band.astype(float)
    
    ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), 0)
    return ndwi

# ============================================================
# COMPOSITE IMAGE LOADING AND CREATION
# ============================================================

def load_composite_image_8_bands(image_path):
    """
    Load composite TIFF and return all 8 bands
    
    Returns:
        numpy array of shape (H, W, 8) or None if error
    """
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        dir_path = os.path.dirname(image_path)
        
        composite_paths = [
            os.path.join(dir_path, f"{base_name}_processed_composite.tiff"),
            os.path.join(dir_path, f"{base_name}_processed_composite.tif"),
            os.path.join(dir_path, f"{base_name}_composite.tiff"),
            os.path.join(dir_path, f"{base_name}_composite.tif")
        ]
        
        composite_path = None
        for path in composite_paths:
            if os.path.exists(path):
                composite_path = path
                break
        
        if composite_path is None:
            print("❌ Composite not found. Tried paths:")
            for p in composite_paths:
                print(f"  - {p}")
            return None
        
        print(f"✅ Loading composite: {composite_path}")
        
        with rasterio.open(composite_path) as src:
            if src.count != 8:
                print(f"⚠️ Warning: Composite has {src.count} bands, expected 8")
                return None
            
            bands = []
            for i in range(1, 9):
                band = src.read(i)
                bands.append(band)
                print(f"   Band {i}: {src.descriptions[i-1] if src.descriptions[i-1] else 'Unknown'} - "
                      f"Shape: {band.shape}, Range: [{band.min():.4f}, {band.max():.4f}]")
            
            composite = np.stack(bands, axis=-1)
            print(f"✅ Composite loaded: shape={composite.shape}, dtype={composite.dtype}")
            return composite
            
    except Exception as e:
        print(f"❌ Error loading composite: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_composite_exists(image_path):
    """Check if 8-band composite exists for given image"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_path = os.path.dirname(image_path)
    
    composite_paths = [
        os.path.join(dir_path, f"{base_name}_processed_composite.tiff"),
        os.path.join(dir_path, f"{base_name}_processed_composite.tif"),
        os.path.join(dir_path, f"{base_name}_composite.tiff"),
        os.path.join(dir_path, f"{base_name}_composite.tif")
    ]
    
    for path in composite_paths:
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    if src.count == 8:
                        return True
            except:
                continue
    
    return False

def calculate_all_indices_for_image(image_path, output_folder):
    """
    Calculate NDSI, NDVI, NDWI and create 8-band composite for single image
    
    Args:
        image_path: Path to input image
        output_folder: Path to output folder
    
    Returns:
        dict with results
    """
    try:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        with rasterio.open(image_path) as src:
            # Detect all required bands
            blue_idx = cached_detect_band(src, 'BLUE')
            green_idx = cached_detect_band(src, 'GREEN')
            red_idx = cached_detect_band(src, 'RED')
            nir_idx = cached_detect_band(src, 'NIR')
            swir_idx = cached_detect_band(src, 'SWIR')
            
            # Check for missing bands
            missing_bands = []
            if not blue_idx: missing_bands.append('BLUE (B2)')
            if not green_idx: missing_bands.append('GREEN (B3)')
            if not red_idx: missing_bands.append('RED (B4)')
            if not nir_idx: missing_bands.append('NIR (B8)')
            if not swir_idx: missing_bands.append('SWIR (B11)')
            
            if missing_bands:
                error_msg = f"Missing bands: {', '.join(missing_bands)}"
                print(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
            
            print(f"✅ All bands detected successfully")
            
            # Read all bands
            blue = src.read(blue_idx).astype(float)
            green = src.read(green_idx).astype(float)
            red = src.read(red_idx).astype(float)
            nir = src.read(nir_idx).astype(float)
            swir = src.read(swir_idx).astype(float)
            
            profile = src.profile.copy()
        
        # Calculate indices
        print("🧮 Calculating NDSI...")
        with np.errstate(divide='ignore', invalid='ignore'):
            ndsi = (green - swir) / (green + swir)
            ndsi = np.where(np.isfinite(ndsi), ndsi, np.nan)
        
        print("🧮 Calculating NDWI...")
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
            ndwi = np.where(np.isfinite(ndwi), ndwi, np.nan)
        
        print("🧮 Calculating NDVI...")
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # ❌ SKIP: Individual index files NOT needed
        # Only creating 8-band composite with all indices included
        
        # print("💾 Saving NDSI...")
        # ndsi_path = os.path.join(output_folder, f"{base_name}_NDSI_processed.tiff")
        # with rasterio.open(ndsi_path, 'w', **single_profile) as dst:
        #     dst.write(ndsi.astype(np.float32), 1)
        #     dst.set_band_description(1, 'NDSI (Snow Index)')
        
        # print("💾 Saving NDWI...")
        # ndwi_path = os.path.join(output_folder, f"{base_name}_NDWI_processed.tiff")
        # with rasterio.open(ndwi_path, 'w', **single_profile) as dst:
        #     dst.write(ndwi.astype(np.float32), 1)
        #     dst.set_band_description(1, 'NDWI (Water Index)')
        
        # print("💾 Saving NDVI...")
        # ndvi_path = os.path.join(output_folder, f"{base_name}_NDVI_processed.tiff")
        # with rasterio.open(ndvi_path, 'w', **single_profile) as dst:
        #     dst.write(ndvi.astype(np.float32), 1)
        #     dst.set_band_description(1, 'NDVI (Vegetation Index)')
        
        # Create 8-band composite
        print("🎨 Creating 8-band composite...")
        composite_path = os.path.join(output_folder, f"{base_name}_processed_composite.tiff")
        
        composite_profile = profile.copy()
        composite_profile.update(
            dtype=rasterio.float32,
            count=8,
            nodata=np.nan
        )
        
        with rasterio.open(composite_path, 'w', **composite_profile) as dst:
            # Write scaled optical bands (divide by 10000 for 0-1 range)
            dst.write((blue / SCALING_FACTOR).astype(np.float32), 1)
            dst.write((green / SCALING_FACTOR).astype(np.float32), 2)
            dst.write((red / SCALING_FACTOR).astype(np.float32), 3)
            dst.write((nir / SCALING_FACTOR).astype(np.float32), 4)
            dst.write((swir / SCALING_FACTOR).astype(np.float32), 5)
            # Write indices as-is (already in -1 to 1 range)
            dst.write(ndsi.astype(np.float32), 6)
            dst.write(ndwi.astype(np.float32), 7)
            dst.write(ndvi.astype(np.float32), 8)
            
            # Set band descriptions
            dst.set_band_description(1, 'BLUE (B2)')
            dst.set_band_description(2, 'GREEN (B3)')
            dst.set_band_description(3, 'RED (B4)')
            dst.set_band_description(4, 'NIR (B8)')
            dst.set_band_description(5, 'SWIR (B11)')
            dst.set_band_description(6, 'NDSI (Snow Index)')
            dst.set_band_description(7, 'NDWI (Water Index)')
            dst.set_band_description(8, 'NDVI (Vegetation Index)')
        
        print(f"✅ 8-band composite created successfully!")
        
        return {
            'success': True,
            'composite_path': composite_path
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================
# GEOTIFF SAVING FUNCTIONS
# ============================================================

def save_geotiff_probability_only(pred_mask, original_img_path, output_path):
    """
    Save ONLY probability .tiff file
    
    Args:
        pred_mask: Probability map (float32, 0.0-1.0)
        original_img_path: Path to original image
        output_path: Output path
    
    Returns:
        str: Path to saved file
    """
    try:
        with rasterio.open(original_img_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'nodata': None
        })
        
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        
        pred_mask_clean = pred_mask.copy()
        pred_mask_clean[~np.isfinite(pred_mask_clean)] = 0.0
        pred_mask_clean = np.clip(pred_mask_clean, 0.0, 1.0)
        
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

def save_ndsi_geotiff(ndsi, original_img_path, output_path, threshold=0.4):
    """Save NDSI result as GeoTIFF"""
    try:
        with rasterio.open(original_img_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
        
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndsi.astype(np.float32), 1)
            dst.set_band_description(1, f'NDSI (threshold={threshold})')
        
        return output_path
    except Exception as e:
        print(f"❌ Error saving NDSI GeoTIFF: {e}")
        raise

# ============================================================
# STATISTICS AND REPORTING
# ============================================================

def generate_prediction_report(pred_mask, binary_mask, original_img_path, output_path_base, 
                                threshold=0.5, processing_time=0.0, model_name="Unknown"):
    """Generate text report with prediction statistics"""
    try:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        
        if pred_mask.max() > 1.0:
            pred_mask_normalized = pred_mask / 255.0
        else:
            pred_mask_normalized = pred_mask
        
        height, width = pred_mask.shape
        total_pixels = height * width
        
        mean_confidence = float(np.mean(pred_mask_normalized))
        std_confidence = float(np.std(pred_mask_normalized))
        min_confidence = float(np.min(pred_mask_normalized))
        max_confidence = float(np.max(pred_mask_normalized))
        median_confidence = float(np.median(pred_mask_normalized))
        
        if binary_mask.max() > 1:
            snow_pixels = int(np.sum(binary_mask > 0))
        else:
            snow_pixels = int(np.sum(binary_mask == 1))
            
        non_snow_pixels = total_pixels - snow_pixels
        snow_percentage = (snow_pixels / total_pixels) * 100
        
        try:
            with rasterio.open(original_img_path) as src:
                crs = str(src.crs) if src.crs else "Unknown"
                bounds = src.bounds
                pixel_area_m2 = abs(src.transform.a * src.transform.e)
                snow_area_km2 = (snow_pixels * pixel_area_m2) / 1_000_000
        except:
            crs = "Unknown"
            bounds = None
            snow_area_km2 = None
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SNOW DETECTION PREDICTION REPORT".center(80))
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("METADATA")
        report_lines.append("-" * 80)
        report_lines.append(f"Generated:        {timestamp}")
        report_lines.append(f"Model Name:       {model_name}")
        report_lines.append(f"Input Image:      {os.path.basename(original_img_path)}")
        report_lines.append(f"Image Size:       {width} x {height} pixels")
        report_lines.append("")
        report_lines.append("PREDICTION STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Mean Confidence:  {mean_confidence:.6f}")
        report_lines.append(f"Snow Pixels:      {snow_pixels:,} ({snow_percentage:.2f}%)")
        if snow_area_km2:
            report_lines.append(f"Snow Coverage:    {snow_area_km2:.4f} km²")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_path = f"{output_path_base}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Report saved: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"⚠️ Could not generate report: {e}")
        return None

# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model_with_composite(model, composite_path, log_callback=None):
    """Validate model with composite file"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    try:
        log("="*80)
        log("MODEL VALIDATION TEST")
        log("="*80)
        
        log(f"\n[LOADING] Composite: {os.path.basename(composite_path)}")
        
        with rasterio.open(composite_path) as src:
            log(f"   Bands: {src.count}")
            log(f"   Size: {src.width}x{src.height}")
            
            if src.count != 8:
                log(f"\n[ERROR] Composite has {src.count} bands, expected 8!")
                return None
            
            composite = np.stack([src.read(i+1) for i in range(8)], axis=-1)
            log(f"   Loaded shape: {composite.shape}")
        
        preprocessed, preprocess_log = preprocess_for_model(composite)
        log(preprocess_log)
        
        patch_size = model.input_shape[1]
        log(f"   Detected patch size: {patch_size}x{patch_size}")
        
        h, w = preprocessed.shape[0], preprocessed.shape[1]
        center_h, center_w = h // 2, w // 2
        h_start = max(0, center_h - patch_size // 2)
        w_start = max(0, center_w - patch_size // 2)
        
        patch = preprocessed[h_start:h_start+patch_size, w_start:w_start+patch_size, :]
        
        batch = np.expand_dims(patch, axis=0)
        prediction = model.predict(batch, verbose=0)
        
        pred_map = prediction[0, :, :, 0]
        mean_conf = np.mean(pred_map)
        
        log(f"\n[RESULT] Mean confidence: {mean_conf:.6f}")
        log("✅ Model validation successful!")
        
        return {'success': True, 'mean_confidence': mean_conf}
        
    except Exception as e:
        log(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None