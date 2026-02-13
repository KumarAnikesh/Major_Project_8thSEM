# ==========================================
# FIXED VERSION - CORRECT MODEL LOADING & SINGLE OUTPUT
# FULL SCREEN + ENHANCED UI + SMART BAND DETECTION
# ==========================================
#
# ✅ FIXES APPLIED (2026-02-11):
#
# 1. CORRECT MODEL LOADING:
#    - dice_loss and dice_coef defined first
#    - Model loaded with custom_objects containing only dice_loss and dice_coef
#    - Removed iou_score from custom objects
#
# 2. SINGLE OUTPUT FILE:
#    - Only saves probability.tif file (float32, 0.0-1.0)
#    - Removed binary_mask.tif, colorized.tif, preview.png
#    - Removed automatic report generation
#
# 3. GEOTIFF OUTPUT:
#    - Preserves geospatial coordinates, CRS, projection
#    - Float32 format for probability values (0.0 to 1.0)
#
# 4. NaN/Inf HANDLING:
#    - Replaces NaN and Inf values with 0
#    - Clips values to valid range [0, 1]
#
# 5. DYNAMIC PATCH SIZE DETECTION:
#    - Auto-detects model input size (128x128 or 256x256)
#    - Updates window_size and stride automatically
#
# OUTPUT FILE PER PREDICTION:
#   1. {name}_prediction_probability.tif - GeoTIFF probability map (0.0-1.0)
#
# COMPATIBILITY:
#   - Cross-platform (Windows, macOS, Linux)
#   - QGIS, ArcGIS, and all GIS software compatible
#   - Handles all data types including NaN/Inf
#   - Works with any model patch size
#
# ==========================================
# ═══════════════════════════════════════════════════════════════════════════
# 🔧 FOLDER PATH FIX APPLIED - Feb 11, 2026
# ═══════════════════════════════════════════════════════════════════════════
#
# FIXED: Folder path clash between NDSI tab and Model Prediction tab
#
# NDSI TAB (Left Column):
#   - Input:  folder_path
#   - Output: output_folder_path
#   - Button: select_output_folder_ndsi()
#
# MODEL TAB (Right Column):
#   - Input:  model_folder_path  
#   - Output: model_output_folder
#   - Button: select_output_folder_model()
#
# These are now COMPLETELY SEPARATE and will not interfere with each other!
#
# ═══════════════════════════════════════════════════════════════════════════



import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import rasterio
import numpy as np
import os
import platform
import matplotlib
matplotlib.use('Agg')  # Cross-platform backend
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage, ImageTk
from tkinter import scrolledtext
import threading

# ============================================================
# CUSTOM OBJECTS FOR MODEL LOADING - CRITICAL FIX
# ============================================================
import tensorflow as tf
import keras.backend as K

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
# MODEL VALIDATION FUNCTIONS (from quick_test.py)
# ============================================================

def preprocess_for_model(composite_image):
    """
    ✅ SUPERVISOR'S APPROACH: NO PREPROCESSING!
    
    Model was trained with RAW data (no normalization, no standardization).
    Just return raw composite as float32 - model handles everything internally.
    
    Returns: (raw_image, log_message)
    """
    print("\n" + "="*80)
    print("🎯 SUPERVISOR'S METHOD: NO PREPROCESSING - USING RAW DATA!")
    print("="*80)
    print("✅ Model handles all preprocessing internally")
    print("✅ No normalization, no standardization, no scaling")
    print("="*80 + "\n")
    
    # Just convert to float32 and handle NaN/Inf
    raw_data = composite_image.astype(np.float32)
    
    # Only replace NaN/Inf with 0 (minimal intervention)
    raw_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
    
    # Print stats for debugging
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
        f"   Mean: {raw_data.mean():.4f}\n"
    )
    
    return raw_data, log_message


# ============================================================
# GEOTIFF SAVING FUNCTION - QGIS-FRIENDLY WITH NaN HANDLING
# ============================================================

def save_geotiff_probability_only(pred_mask, original_img_path, output_path):
    """
    ✅ FIXED: Save ONLY probability .tiff file
    
    Args:
        pred_mask: Probability map (float32, 0.0-1.0)
        original_img_path: Path to original image (to get CRS and transform)
        output_path: Full path for output probability.tiff file
    
    Returns:
        str: Path to saved probability file
    """
    try:
        # Open original image to get geospatial metadata
        with rasterio.open(original_img_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            
        # Update profile for output
        profile.update({
            'count': 1,
            'dtype': 'float32',  # Keep as float32 for probability values
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'nodata': None
        })
        
        # Ensure mask is 2D
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        
        # Handle NaN and Inf
        pred_mask_clean = pred_mask.copy()
        pred_mask_clean[~np.isfinite(pred_mask_clean)] = 0.0
        
        # Clip to valid range [0, 1]
        pred_mask_clean = np.clip(pred_mask_clean, 0.0, 1.0)
        
        # Save probability map as float32 (0.0 to 1.0)
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


# ============================================================
# PROFESSIONAL PREDICTION REPORT GENERATOR
# ============================================================

def generate_prediction_report(pred_mask, binary_mask, original_img_path, output_path_base, 
                                threshold=0.5, processing_time=0.0, model_name="Unknown"):
    """Generate professional TXT report with prediction statistics"""
    from datetime import datetime
    
    try:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        
        # Normalize pred_mask to 0-1
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
        
        very_low = int(np.sum(pred_mask_normalized < 0.2))
        low = int(np.sum((pred_mask_normalized >= 0.2) & (pred_mask_normalized < 0.4)))
        medium = int(np.sum((pred_mask_normalized >= 0.4) & (pred_mask_normalized < 0.6)))
        high = int(np.sum((pred_mask_normalized >= 0.6) & (pred_mask_normalized < 0.8)))
        very_high = int(np.sum(pred_mask_normalized >= 0.8))
        
        try:
            with rasterio.open(original_img_path) as src:
                crs = str(src.crs) if src.crs else "Unknown"
                bounds = src.bounds
                pixel_area_m2 = abs(src.transform.a * src.transform.e)
                snow_area_km2 = (snow_pixels * pixel_area_m2) / 1_000_000
                total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000
        except:
            crs = "Unknown"
            bounds = None
            snow_area_km2 = None
        
        snow_detected = "YES" if snow_pixels > 0 else "NO"
        confidence_level = "HIGH" if mean_confidence > 0.7 else "MEDIUM" if mean_confidence > 0.4 else "LOW"
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
        report_lines.append(f"Total Pixels:     {total_pixels:,}")
        report_lines.append(f"Coordinate System: {crs}")
        if bounds:
            report_lines.append(f"Bounds:           W={bounds.left:.6f}, E={bounds.right:.6f}")
            report_lines.append(f"                  S={bounds.bottom:.6f}, N={bounds.top:.6f}")
        if processing_time > 0:
            report_lines.append(f"Processing Time:  {processing_time:.2f} seconds")
        report_lines.append("")
        
        report_lines.append("PREDICTION STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Mean Confidence:  {mean_confidence:.6f}")
        report_lines.append(f"Std Deviation:    {std_confidence:.6f}")
        report_lines.append(f"Min Probability:  {min_confidence:.6f}")
        report_lines.append(f"Max Probability:  {max_confidence:.6f}")
        report_lines.append(f"Median:           {median_confidence:.6f}")
        report_lines.append("")
        
        report_lines.append("CLASSIFICATION RESULTS")
        report_lines.append("-" * 80)
        report_lines.append(f"Snow Pixels:      {snow_pixels:,} ({snow_percentage:.2f}%)")
        report_lines.append(f"Non-snow Pixels:  {non_snow_pixels:,} ({100-snow_percentage:.2f}%)")
        if snow_area_km2:
            report_lines.append(f"Snow Coverage:    {snow_area_km2:.4f} km²")
        report_lines.append("")
        
        report_lines.append("CONFIDENCE DISTRIBUTION")
        report_lines.append("-" * 80)
        report_lines.append(f"Very Low  (0.0-0.2): {very_low:,} ({very_low/total_pixels*100:.2f}%)")
        report_lines.append(f"Low       (0.2-0.4): {low:,} ({low/total_pixels*100:.2f}%)")
        report_lines.append(f"Medium    (0.4-0.6): {medium:,} ({medium/total_pixels*100:.2f}%)")
        report_lines.append(f"High      (0.6-0.8): {high:,} ({high/total_pixels*100:.2f}%)")
        report_lines.append(f"Very High (0.8-1.0): {very_high:,} ({very_high/total_pixels*100:.2f}%)")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("VERDICT".center(80))
        report_lines.append("=" * 80)
        report_lines.append(("✅ SUCCESS - Snow detected!" if snow_detected == "YES" else "⚠️  No snow detected").center(80))
        report_lines.append("")
        report_lines.append(f"Snow Detected:     {snow_detected}")
        report_lines.append(f"Confidence Level:  {confidence_level}")
        report_lines.append(f"Mean Confidence:   {mean_confidence:.6f}")
        report_lines.append(f"Prediction Range:  {min_confidence:.6f} to {max_confidence:.6f}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_path = f"{output_path_base}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   📄 Report: {os.path.basename(report_path)}")
        return report_path
        
    except Exception as e:
        print(f"⚠️ Could not generate report: {e}")
        return None


def validate_model_with_composite(model, composite_path, log_callback=None):
    """
    Validate a model by running prediction on a composite file.
    Returns: dict with validation results or None if failed
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
    
    try:
        log("="*80)
        log("MODEL VALIDATION TEST")
        log("="*80)
        
        # Load composite
        log(f"\n[LOADING] Composite: {os.path.basename(composite_path)}")
        
        with rasterio.open(composite_path) as src:
            log(f"   Bands: {src.count}")
            log(f"   Size: {src.width}x{src.height}")
            log(f"   Dtype: {src.dtypes[0]}")
            
            if src.count != 8:
                log(f"\n[ERROR] Composite has {src.count} bands, expected 8!")
                log("   Please regenerate composite using 'All Indices' in GUI")
                return None
            
            # Read all 8 bands
            composite = np.stack([src.read(i+1) for i in range(8)], axis=-1)
            log(f"   Loaded shape: {composite.shape}")
            log(f"   Value range: [{composite.min():.6f}, {composite.max():.6f}]")
        
        # Apply preprocessing
        log("\n" + "="*80)
        preprocessed, preprocess_log = preprocess_for_model(composite)
        log(preprocess_log)
        log("="*80)
        
        # Check model input shape
        log(f"\n[MODEL INFO]")
        log(f"   Input shape: {model.input_shape}")
        log(f"   Output shape: {model.output_shape}")
        
        expected_channels = model.input_shape[3]
        if expected_channels != 8:
            log(f"\n[WARNING] Model expects {expected_channels} channels, but composite has 8!")
        
        # ✅ DYNAMIC PATCH SIZE - Auto-detect from model input shape
        patch_size = model.input_shape[1]  # Extract patch size from model (128 or 256)
        log(f"   Detected patch size: {patch_size}x{patch_size}")
        
        # Test prediction on patch
        log("\n" + "="*80)
        log(f"TESTING PREDICTION ON {patch_size}x{patch_size} PATCH")
        log("="*80)
        
        # Extract center patch
        h, w = preprocessed.shape[0], preprocessed.shape[1]
        center_h, center_w = h // 2, w // 2
        h_start = max(0, center_h - patch_size // 2)
        w_start = max(0, center_w - patch_size // 2)
        h_end = min(h, h_start + patch_size)
        w_end = min(w, w_start + patch_size)
        
        patch = preprocessed[h_start:h_end, w_start:w_end, :]
        
        # Pad if needed
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size, 8), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded
        
        log(f"Patch location: [{h_start}:{h_end}, {w_start}:{w_end}]")
        log(f"Patch shape: {patch.shape}")
        
        # Add batch dimension
        batch = np.expand_dims(patch, axis=0)
        log(f"Batch shape: {batch.shape}")
        
        # Predict
        log("\n[RUNNING] Prediction...")
        prediction = model.predict(batch, verbose=0)
        
        log(f"Prediction shape: {prediction.shape}")
        
        # Analyze prediction
        pred_map = prediction[0, :, :, 0]
        
        log(f"\n[STATISTICS] Prediction Statistics:")
        mean_conf = np.mean(pred_map)
        std_conf = np.std(pred_map)
        min_conf = np.min(pred_map)
        max_conf = np.max(pred_map)
        median_conf = np.median(pred_map)
        
        log(f"   Mean confidence: {mean_conf:.6f}")
        log(f"   Std deviation: {std_conf:.6f}")
        log(f"   Min: {min_conf:.6f}")
        log(f"   Max: {max_conf:.6f}")
        log(f"   Median: {median_conf:.6f}")
        
        # Classify pixels
        threshold = 0.5
        snow_pixels = np.sum(pred_map > threshold)
        total_pixels = pred_map.size
        snow_percent = (snow_pixels / total_pixels) * 100
        
        log(f"\n[CLASSIFICATION] Using threshold={threshold}:")
        log(f"   Snow pixels: {snow_pixels:,} ({snow_percent:.2f}%)")
        log(f"   Non-snow pixels: {total_pixels - snow_pixels:,} ({100-snow_percent:.2f}%)")
        
        # Verdict
        log("\n" + "="*80)
        log("VERDICT")
        log("="*80)
        
        if max_conf < 0.01:
            verdict_status = "problem"
            verdict_message = "[PROBLEM] Model outputs near-zero everywhere!"
            log(verdict_message)
            log(f"   Max confidence: {max_conf:.6f}")
            log(f"   Mean confidence: {mean_conf:.6f}")
        elif max_conf < 0.3:
            verdict_status = "warning"
            verdict_message = "[WARNING] Low confidence predictions"
            log(verdict_message)
            log(f"   Max confidence: {max_conf:.6f}")
            log(f"   Mean confidence: {mean_conf:.6f}")
        else:
            verdict_status = "success"
            verdict_message = "[SUCCESS] Model is producing predictions!"
            log(verdict_message)
            log(f"   Max confidence: {max_conf:.6f}")
            log(f"   Mean confidence: {mean_conf:.6f}")
            log(f"   Snow coverage: {snow_percent:.1f}%")
        
        log("="*80)
        
        return {
            'status': verdict_status,
            'message': verdict_message,
            'stats': {
                'mean': mean_conf,
                'std': std_conf,
                'min': min_conf,
                'max': max_conf,
                'median': median_conf,
                'snow_percent': snow_percent
            }
        }
        
    except Exception as e:
        log(f"\n[ERROR] Validation failed: {e}")
        import traceback
        log(traceback.format_exc())
        return None


def open_validation_window():
    """Open a new window for model validation"""
    global validation_window, validation_text_widget, root
    
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please load a model first!")
        return
    
    # Create validation window
    validation_window = tk.Toplevel(root)
    validation_window.title("Model Validation - Quick Test")
    validation_window.geometry("900x700")
    validation_window.configure(bg=COLORS['background'])
    
    # Header
    header_frame = tk.Frame(validation_window, bg=COLORS['card_bg'], height=60)
    header_frame.pack(fill='x', padx=15, pady=(15, 10))
    header_frame.pack_propagate(False)
    
    tk.Label(
        header_frame,
        text="🔍 Model Validation Test",
        font=("Segoe UI", 18, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text']
    ).pack(pady=15)
    
    # Info label
    info_frame = tk.Frame(validation_window, bg=COLORS['card_bg'])
    info_frame.pack(fill='x', padx=15, pady=(0, 10))
    
    tk.Label(
        info_frame,
        text=f"Model: {os.path.basename(model_path) if model_path else 'Unknown'}",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w',
        padx=15,
        pady=8
    ).pack(fill='x')
    
    # Select composite button
    btn_frame = tk.Frame(validation_window, bg=COLORS['background'])
    btn_frame.pack(fill='x', padx=15, pady=(0, 10))
    
    tk.Button(
        btn_frame,
        text="📁  Select Composite for Validation",
        font=_FONT_BTN,
        bg=COLORS['primary'],
        fg='#ffffff',
        activebackground=COLORS['primary_hover'],
        activeforeground='#ffffff',
        pady=12,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=run_validation_test
    ).pack(fill='x')
    
    # Log display
    log_frame = tk.Frame(validation_window, bg=COLORS['card_bg'])
    log_frame.pack(fill='both', expand=True, padx=15, pady=(0, 10))
    
    tk.Label(
        log_frame,
        text="Validation Log:",
        font=("Segoe UI", 11, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w',
        padx=10,
        pady=8
    ).pack(fill='x')
    
    # Scrolled text for logs
    validation_text_widget = scrolledtext.ScrolledText(
        log_frame,
        font=("Consolas", 9) if IS_WINDOWS else ("Monaco", 9),
        bg=COLORS['details_bg'],
        fg=COLORS['text'],
        insertbackground=COLORS['text'],
        relief='flat',
        bd=0,
        padx=10,
        pady=10,
        wrap='word'
    )
    validation_text_widget.pack(fill='both', expand=True, padx=10, pady=(0, 10))
    
    # Initial message
    validation_text_widget.insert('1.0', 
        "Ready to validate model.\n\n"
        "Click 'Select Composite for Validation' to choose an 8-band composite file.\n"
        "The validation will:\n"
        "  1. Load the composite\n"
        "  2. Apply correct preprocessing\n"
        "  3. Run prediction on a test patch\n"
        "  4. Analyze model output\n"
        "  5. Provide verdict on model compatibility\n"
    )
    validation_text_widget.config(state='disabled')
    
    # Close button
    tk.Button(
        validation_window,
        text="Close",
        font=_FONT_BTN,
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=10,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=validation_window.destroy
    ).pack(fill='x', padx=15, pady=(0, 15))


def run_validation_test():
    """Run the validation test on a selected composite"""
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please load a model first!")
        return
    
    # Select composite file
    composite_path = filedialog.askopenfilename(
        title="Select 8-Band Composite for Validation",
        filetypes=[
            ("TIFF files", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not composite_path:
        return
    
    # Clear previous log
    if validation_text_widget:
        validation_text_widget.config(state='normal')
        validation_text_widget.delete('1.0', 'end')
        validation_text_widget.config(state='disabled')
    
    # Define log callback
    def log_message(msg):
        if validation_text_widget:
            validation_text_widget.config(state='normal')
            validation_text_widget.insert('end', msg + '\n')
            validation_text_widget.see('end')
            validation_text_widget.config(state='disabled')
            validation_text_widget.update()
    
    # Run validation in a thread to keep UI responsive
    def run_in_thread():
        result = validate_model_with_composite(loaded_model, composite_path, log_message)
        
        if result:
            status = result['status']
            stats = result['stats']
            
            if status == 'success':
                messagebox.showinfo(
                    "✓ Validation Successful",
                    f"{result['message']}\n\n"
                    f"Max Confidence: {stats['max']:.4f}\n"
                    f"Mean Confidence: {stats['mean']:.4f}\n"
                    f"Snow Coverage: {stats['snow_percent']:.1f}%\n\n"
                    "Model is working correctly!"
                )
            elif status == 'warning':
                messagebox.showwarning(
                    "⚠️ Low Confidence",
                    f"{result['message']}\n\n"
                    f"Max Confidence: {stats['max']:.4f}\n"
                    f"Mean Confidence: {stats['mean']:.4f}"
                )
            else:
                messagebox.showerror(
                    "❌ Validation Issue",
                    f"{result['message']}\n\n"
                    f"Max Confidence: {stats['max']:.6f}\n"
                    f"Mean Confidence: {stats['mean']:.6f}"
                )
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()



# Platform detection for compatibility
PLATFORM_SYSTEM = platform.system()
IS_WINDOWS = PLATFORM_SYSTEM == 'Windows'
IS_MAC = PLATFORM_SYSTEM == 'Darwin'
IS_LINUX = PLATFORM_SYSTEM == 'Linux'

# ---------------- GLOBALS (🔟 Centralised at top) ----------------
image_path = None
result_data = None
profile = None
green = None
swir = None
ndsi_threshold = 0.4
folder_path = None
image_files = []
current_image_index = 0
output_folder_path = None
loaded_model = None
model_path = None
_band_cache = {}          # 1️⃣1️⃣ band-detection cache  {filepath: {band_type: index}}
SCALING_FACTOR = 10000    # Scale factor for composite image bands

# NEW: Model inference globals
model_input_image_path = None
model_prediction_result = None
model_preview_canvas = None
model_result_label = None
model_batch_folder = None
model_output_folder = None

# ✨ NEW: Unified prediction mode variables (NDSI tab style)
model_prediction_mode = None  # StringVar: "single" or "batch"
model_folder_path = None  # Folder containing images (like NDSI tab's folder_path)
model_image_files = []  # List of image files in folder (like NDSI tab's image_files)
model_current_image_index = 0  # Currently selected image index
model_image_listbox = None  # Listbox widget reference
model_selected_image_listbox = None  # Right column selected image listbox
model_folder_label = None  # Folder display label
model_folder_info_label = None  # Folder info label
model_file_name_label = None  # Selected file name label
model_file_info_label = None  # Selected file info label
model_output_folder_label = None  # Output folder display label (NEW)
model_output_folder_info = None  # Output folder info label (NEW)

# ✅ DYNAMIC: Will be set based on loaded model's input shape
window_size = 128  # Default window size (updated when model is loaded)
stride = 64  # Default stride (overlap = window_size - stride)

# GUI widget variables (will be set when GUI is created)
window_size_var = None
stride_var = None

# NEW: Validation window globals
validation_window = None
validation_text_widget = None

# ---------------- CONSISTENT COLOR PALETTE (2️⃣) ----------------
COLORS = {
    # Core surfaces
    'background':       '#0f1117',   # Deep charcoal – entire window
    'card_bg':          '#1a1d27',   # Slightly lighter – every card
    'card_border':      '#2a2e3a',   # Subtle card outline

    # Primary accent (buttons, highlights, active states)
    'primary':          '#5b8cfa',   # Bright periwinkle blue
    'primary_hover':    '#7aa3fc',
    'primary_disabled': '#3a4155',   # Muted, clearly inactive

    # Secondary surfaces (folder buttons, upload model)
    'secondary_bg':     '#2e3347',
    'secondary_hover':  '#3b4260',
    'secondary_border': '#5a6478',

    # Text
    'text':             '#e8eaf0',   # High contrast on dark
    'text_muted':       '#6b7280',   # Info / placeholder
    'text_link':        '#5b8cfa',   # Same as primary

    # Feedback
    'success':          '#4ade80',
    'warning':          '#fbbf24',
    'error':            '#f87171',

    # Preview canvas
    'canvas_bg':        '#111827',   # Dark so raster data pops (4️⃣)

    # Listbox / combobox
    'listbox_bg':       '#141620',
    'listbox_sel':      '#5b8cfa',

    # Model details text area
    'details_bg':       '#141620',
}

# Derive platform-appropriate font families once
_FONT_TITLE   = ("SF Pro Display", 26, "bold") if IS_MAC else ("Segoe UI",    26, "bold")   # 5️⃣ Title
_FONT_SUB     = ("SF Pro Display", 11)         if IS_MAC else ("Segoe UI",    11)            # Subtitle
_FONT_SECTION = ("SF Pro Display", 12, "bold") if IS_MAC else ("Segoe UI",    12, "bold")   # Section headers
_FONT_BODY    = ("SF Pro Display", 10)         if IS_MAC else ("Segoe UI",    10)            # Normal body
_FONT_SMALL   = ("SF Pro Display", 9)          if IS_MAC else ("Segoe UI",     9)            # Info / muted
_FONT_BTN     = ("SF Pro Display", 11, "bold") if IS_MAC else ("Segoe UI",    11, "bold")   # Buttons
_FONT_BADGE   = ("SF Pro Display", 10, "bold") if IS_MAC else ("Segoe UI",    10, "bold")   # Threshold badge


# ============================================================
# 6️⃣  CENTRALISED STATUS HELPER
# ============================================================
_status_widget = None   # assigned after widget is created

def set_status(text, level="info"):
    """level: 'success' | 'warning' | 'error' | 'info'"""
    color_map = {
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error':   COLORS['error'],
        'info':    COLORS['text'],
    }
    if _status_widget:
        _status_widget.config(text=text, fg=color_map.get(level, COLORS['text']))


# ============================================================
# 7️⃣  RUN-BUTTON ENABLE / DISABLE HELPER
# ============================================================
_btn_run = None   # assigned after widget is created

def _update_run_button():
    """Enable Run only when an image is loaded AND has >= 2 bands."""
    if _btn_run is None:
        return
    if image_path and os.path.isfile(image_path):
        try:
            with rasterio.open(image_path) as src:
                if src.count >= 2:
                    _btn_run.config(state='normal',
                                    bg=COLORS['primary'],
                                    activebackground=COLORS['primary_hover'],
                                    highlightbackground='#7aa3fc',
                                    highlightcolor='#7aa3fc')
                    return
        except Exception:
            pass
    # Fall-through → disabled
    _btn_run.config(state='disabled',
                    bg=COLORS['primary_disabled'],
                    activebackground=COLORS['primary_disabled'],
                    highlightbackground='#4a5568',
                    highlightcolor='#4a5568')


# ---------------- SMART BAND DETECTION ----------------
def detect_band_by_type(src, band_type):
    """
    Detect band by its type using metadata/descriptions
    Works with GEE exports and standard satellite images

    Args:
        src: rasterio dataset
        band_type: 'GREEN', 'RED', 'NIR', 'SWIR'

    Returns:
        band_index (1-based) or None
    """
    # FIX: Check if dataset is valid before accessing to prevent GDAL NULL pointer errors
    if src is None or src.closed:
        return None
    
    try:
        band_descriptions = src.descriptions
        band_count = src.count
    except Exception:
        return None

    # Patterns to match for each band type
    patterns = {
        'GREEN': ['B3', 'GREEN', 'GRN', 'BAND3', 'BAND 3', 'SR_B3'],
        'RED':   ['B4', 'RED', 'BAND4', 'BAND 4', 'SR_B4'],
        'NIR':   ['B8', 'NIR', 'BAND8', 'BAND 8', 'B8A', 'SR_B8', 'NEAR_INFRARED'],
        'SWIR':  ['B11', 'SWIR', 'SWIR1', 'BAND11', 'BAND 11', 'SR_B11'],
        'BLUE':  ['B2', 'BLUE', 'BLU', 'BAND2', 'BAND 2', 'SR_B2']
    }

    if band_type not in patterns:
        return None

    # Method 1: Check band descriptions (most reliable)
    if band_descriptions:
        for i, desc in enumerate(band_descriptions):
            if desc:
                desc_upper = str(desc).upper().strip()

                # Exact match first
                for pattern in patterns[band_type]:
                    if desc_upper == pattern.upper():
                        return i + 1

                # Partial match
                for pattern in patterns[band_type]:
                    if pattern.upper() in desc_upper:
                        return i + 1

    return None


def _cached_detect_band(src, band_type):
    """1️⃣1️⃣ Wrapper – returns cached result when available."""
    global _band_cache
    
    # FIX: Check if dataset is valid to prevent GDAL errors
    if src is None or src.closed:
        return None
    
    try:
        key = src.name          # file path is unique per dataset
        if key not in _band_cache:
            _band_cache[key] = {}
        cache = _band_cache[key]
        if band_type not in cache:
            cache[band_type] = detect_band_by_type(src, band_type)
        return cache[band_type]
    except Exception:
        # If anything fails, try direct detection
        return detect_band_by_type(src, band_type)


def get_required_bands(src, index_type):
    """
    Get required bands for a specific index calculation

    Args:
        src: rasterio dataset
        index_type: 'NDSI', 'NDWI', 'NDVI'

    Returns:
        dict with band data and info
    """
    result = {'success': False, 'bands': {}, 'info': '', 'missing': []}

    if index_type == 'NDSI':
        required = ['GREEN', 'SWIR']
    elif index_type == 'NDWI':
        required = ['GREEN', 'NIR']
    elif index_type == 'NDVI':
        required = ['NIR', 'RED']
    else:
        result['info'] = f"Unknown index type: {index_type}"
        return result

    # Detect each required band
    band_info_list = []
    for band_type in required:
        band_idx = _cached_detect_band(src, band_type)  # 1️⃣1️⃣ cached

        if band_idx:
            band_data = src.read(band_idx).astype(float)
            result['bands'][band_type] = {
                'data': band_data,
                'index': band_idx,
                'name': src.descriptions[band_idx-1] if src.descriptions and src.descriptions[band_idx-1] else f"Band {band_idx}"
            }
            band_info_list.append(f"{band_type}: {result['bands'][band_type]['name']}")
        else:
            result['missing'].append(band_type)

    # Check if all required bands found
    if not result['missing']:
        result['success'] = True
        result['info'] = f"Using {' & '.join(band_info_list)}"
    else:
        result['info'] = f"Missing bands: {', '.join(result['missing'])}"

    return result


# ---------------- 🔧 FIXED: LOAD 8-BAND COMPOSITE ----------------
def load_composite_image_8_bands(image_path):
    """
    Load composite TIFF and return all 8 bands properly
    
    Args:
        image_path: Path to original image (will look for _composite.tif)
    
    Returns:
        numpy array of shape (H, W, 8) or None if error
        
    Composite band order:
        Band 1: Blue (B2) / 10000
        Band 2: Green (B3) / 10000  
        Band 3: Red (B4) / 10000
        Band 4: NIR (B8) / 10000
        Band 5: SWIR (B11) / 10000
        Band 6: NDSI (Snow Index)
        Band 7: NDWI (Water Index)
        Band 8: NDVI (Vegetation Index)
    """
    try:
        # Get base name and directory without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        dir_path = os.path.dirname(image_path)
        
        # Try different composite path patterns
        # Use base_name to avoid .replace() duplicating suffixes
        composite_paths = [
            os.path.join(dir_path, f"{base_name}_composite.tiff"),
            os.path.join(dir_path, f"{base_name}_composite.tif"),
            os.path.join(dir_path, f"{base_name}_processed_composite.tiff"),
            os.path.join(dir_path, f"{base_name}_processed_composite.tif")
        ]
        
        composite_path = None
        for path in composite_paths:
            if os.path.exists(path):
                composite_path = path
                break
        
        if composite_path is None:
            print("Composite not found. Tried paths:")
            for p in composite_paths:
                print(f"  - {p}")
            return None
        
        print(f"Loading composite: {composite_path}")
        
        with rasterio.open(composite_path) as src:
            if src.count != 8:
                print(f"Warning: Composite has {src.count} bands, expected 8")
                print(f"Band descriptions: {src.descriptions}")
                return None
            
            # Read all 8 bands in correct order
            bands = []
            for i in range(1, 9):
                band = src.read(i)
                bands.append(band)
                print(f"Band {i}: {src.descriptions[i-1] if src.descriptions[i-1] else 'Unknown'} - "
                      f"Shape: {band.shape}, Range: [{band.min():.4f}, {band.max():.4f}]")
            
            # Stack into (H, W, 8)
            composite = np.stack(bands, axis=-1)
            
            print(f"✓ Composite loaded: shape={composite.shape}, dtype={composite.dtype}")
            return composite
            
    except Exception as e:
        print(f"Error loading composite: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------- THRESHOLD HANDLER ----------------
def on_threshold_change(val):
    global ndsi_threshold
    ndsi_threshold = float(val)
    # 8️⃣ badge updates automatically
    threshold_value_label.config(text=f"{ndsi_threshold:.2f}")


# ---------------- MODEL PREDICTION FUNCTIONS ----------------

# ---------------- MODEL INFERENCE FUNCTIONS (IMPROVED) ----------------

def check_composite_exists(image_path):
    """
    Check if the 8-band composite exists for the given image
    
    Args:
        image_path: Path to the original image
    
    Returns:
        bool: True if composite exists, False otherwise
    """
    # Get base name and directory without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_path = os.path.dirname(image_path)
    
    # Try different composite path patterns
    # Use base_name to avoid .replace() duplicating suffixes
    composite_paths = [
        os.path.join(dir_path, f"{base_name}_composite.tiff"),
        os.path.join(dir_path, f"{base_name}_composite.tif"),
        os.path.join(dir_path, f"{base_name}_processed_composite.tiff"),
        os.path.join(dir_path, f"{base_name}_processed_composite.tif")
    ]
    
    for path in composite_paths:
        if os.path.exists(path):
            # Verify it has 8 bands
            try:
                with rasterio.open(path) as src:
                    if src.count == 8:
                        return True
            except:
                continue
    
    return False


# Global variable to hold prediction button reference
_prediction_button = None
_composite_status_label = None
_stored_model_details = ""  # Store model details for popup window

def update_prediction_button_state(has_composite):
    """Update the prediction button appearance based on composite availability"""
    global _prediction_button, _composite_status_label
    
    if _prediction_button:
        if has_composite:
            _prediction_button.config(
                text="▶  Run Prediction",
                bg=COLORS['primary']
            )
        else:
            _prediction_button.config(
                text="⚠️  Run Prediction (No Composite!)",
                bg=COLORS['warning']
            )
    
    if _composite_status_label:
        if has_composite:
            _composite_status_label.config(
                text="✓ 8-band composite detected",
                fg=COLORS['success']
            )
        else:
            _composite_status_label.config(
                text="⚠️ Missing composite - Run 'All Indices' first!",
                fg=COLORS['warning']
            )




# ============================================================
# UNIFIED MODEL PREDICTION HELPERS (NDSI-style interface)
# ============================================================

def upload_model_folder():
    """
    Upload folder containing images for model prediction (batch or single selection).
    Similar to NDSI tab's upload_folder() function.
    """
    global model_folder_path, model_image_files, model_current_image_index, model_input_image_path, model_prediction_result, model_selected_image_listbox
    
    folder = filedialog.askdirectory(title="Select Folder Containing Images for Prediction")
    
    if not folder:
        return
    
    # Find all TIFF files in the folder
    model_image_files = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.tif', '.tiff')) and 'composite' in file.lower():
            model_image_files.append(os.path.join(folder, file))
    
    model_image_files.sort()  # Sort alphabetically
    
    if not model_image_files:
        messagebox.showwarning("No Images Found", "No GeoTIFF files found in the selected folder.")
        model_folder_path = None
        return
    
    model_folder_path = folder
    model_current_image_index = 0
    model_prediction_result = None
    
    # Update UI
    if model_folder_label:
        model_folder_label.config(
            text=f"{os.path.basename(folder)}",
            fg=COLORS['success']
        )
    
    if model_folder_info_label:
        model_folder_info_label.config(
            text=f"{len(model_image_files)} GeoTIFF images found",
            fg=COLORS['text_muted']
        )
    
    # Populate listbox (left column)
    if model_image_listbox:
        model_image_listbox.delete(0, tk.END)
        for img_file in model_image_files:
            model_image_listbox.insert(tk.END, os.path.basename(img_file))
        
        # Select first image - this will automatically update the right column via load_selected_model_image()
        if model_image_files:
            model_image_listbox.selection_set(0)
            load_selected_model_image()
    
    set_status(f"✓ Model folder loaded: {len(model_image_files)} images found – Select image or run batch prediction", "success")


def load_selected_model_image(event=None):
    """
    Load the selected image from the listbox for single prediction.
    Updates the right column listbox with selected image.
    """
    global model_input_image_path, model_current_image_index, model_prediction_result, model_selected_image_listbox
    
    print(f"DEBUG: load_selected_model_image called")
    print(f"DEBUG: model_selected_image_listbox is: {model_selected_image_listbox}")
    
    if not model_image_listbox:
        print(f"DEBUG: model_image_listbox is None, returning")
        return
    
    selection = model_image_listbox.curselection()
    
    if not selection:
        print(f"DEBUG: No selection, returning")
        return
    
    model_current_image_index = selection[0]
    model_input_image_path = model_image_files[model_current_image_index]
    model_prediction_result = None
    
    print(f"DEBUG: Selected image: {os.path.basename(model_input_image_path)}")
    
    try:
        with rasterio.open(model_input_image_path) as src:
            band_count = src.count
            width = src.width
            height = src.height
        
        if model_file_name_label:
            model_file_name_label.config(
                text=f"Current: {os.path.basename(model_input_image_path)}",
                fg='#00FF00'  # Green color for single selection
            )
        
        if model_file_info_label:
            model_file_info_label.config(
                text=f"{band_count} bands | {width}x{height} pixels",
                fg=COLORS['text_muted']
            )
        
        # Check if file is a valid 8-band composite
        # Simple check: if band_count == 8, it's a composite file
        if band_count == 8:
            composite_exists = True  # This IS a valid 8-band composite
        else:
            composite_exists = False  # Not a composite, need to find one
            # Only try to find composite if this is a processed file
            if 'processed' in os.path.basename(model_input_image_path).lower():
                composite_exists = check_composite_exists(model_input_image_path)
        
        update_prediction_button_state(composite_exists)
        
        # Update the selected image listbox (right column) - show only selected image
        print(f"DEBUG: About to update model_selected_image_listbox")
        print(f"DEBUG: model_selected_image_listbox type: {type(model_selected_image_listbox)}")
        
        if model_selected_image_listbox:
            print(f"DEBUG: Clearing and inserting into listbox")
            model_selected_image_listbox.delete(0, tk.END)
            model_selected_image_listbox.insert(0, f"✓ {os.path.basename(model_input_image_path)}")
            print(f"DEBUG: Successfully updated listbox")
        else:
            print(f"DEBUG: model_selected_image_listbox is None!")
        
        if composite_exists:
            set_status(f"✓ Image {model_current_image_index + 1}/{len(model_image_files)} loaded (Composite found)", "success")
        else:
            set_status(f"⚠️ Image loaded but NO COMPOSITE - Run 'All Indices' first!", "warning")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        set_status("❌ Error loading image", "error")


def clear_model_image_selection():
    """
    Clear the image selection in the listbox (enables batch mode).
    Shows "ALL IMAGES SELECTED(Batch Mode)" in yellow.
    """
    global model_input_image_path
    
    if model_image_listbox:
        model_image_listbox.selection_clear(0, tk.END)
    
    model_input_image_path = None
    
    if model_file_name_label:
        model_file_name_label.config(
            text="ALL IMAGES SELECTED (Batch Mode)",
            fg='#FFD700'  # Yellow/Gold color
        )
    
    if model_file_info_label:
        model_file_info_label.config(
            text="Run 'All Indices' to process all images in folder",
            fg=COLORS['text_muted']
        )
        
    
    # Show all images in the selected image listbox (batch mode)
    if model_selected_image_listbox and model_image_files:
        model_selected_image_listbox.delete(0, tk.END)
        for img_file in model_image_files:
            model_selected_image_listbox.insert(tk.END, f"📄 {os.path.basename(img_file)}")
    
    set_status("✓ Selection cleared – Batch mode enabled", "info")


# ============================================================
# ORIGINAL MODEL FUNCTIONS (kept for backwards compatibility)
# ============================================================

def select_model_input_image():

    """Select an image for model inference"""
    global model_input_image_path
    
    filepath = filedialog.askopenfilename(
        title="Select Image for Model Prediction",
        filetypes=[
            ("Image Files", "*.tif *.tiff *.png *.jpg *.jpeg"),
            ("TIFF Files", "*.tif *.tiff"),
            ("All Files", "*.*")
        ]
    )
    
    if not filepath:
        return
    
    model_input_image_path = filepath
    
    # ✨ NEW: Check if composite exists and warn user
    composite_exists = check_composite_exists(filepath)
    
    # Update button state
    update_prediction_button_state(composite_exists)
    
    # Display preview
    display_model_input_preview(filepath)
    
    if composite_exists:
        set_status(f"✓ Image selected: {os.path.basename(filepath)} (Composite found)", "success")
    else:
        set_status(f"⚠️ Image selected but NO COMPOSITE found! Run 'All Indices' first.", "warning")
        
        # Show helpful warning dialog
        messagebox.showwarning(
            "⚠️ Composite Image Missing",
            f"Selected: {os.path.basename(filepath)}\n\n"
            "❌ No 8-band composite found!\n\n"
            "The model requires a preprocessed composite image with:\n"
            "• Bands 1-5: Blue, Green, Red, NIR, SWIR\n"
            "• Bands 6-8: NDSI, NDWI, NDVI\n\n"
            "📋 TO FIX THIS:\n"
            "1. Go to 'NDSI Calculation' tab\n"
            "2. Load this image\n"
            "3. Click 'Run All Indices'\n"
            "4. Come back and run prediction\n\n"
            "⚠️ Without the composite, predictions will FAIL or produce incorrect results!"
        )


def display_model_input_preview(img_path):
    """
    Display the selected image in model preview canvas
    NOTE: This function is now disabled as preview canvas has been removed.
    The image list is displayed in the listbox instead.
    """
    # Function disabled - preview canvas removed
    pass



import numpy as np
from PIL import Image as PILImage
import rasterio
import os

def predict_with_sliding_window(model, img_data, window_size, stride, target_channels):
    """
    Apply model prediction using sliding window technique
    
    Args:
        model: Loaded Keras model
        img_data: Original image data (H, W, C) - MUST be preprocessed composite
        window_size: Size of sliding window (assumes square window)
        stride: Step size for sliding window
        target_channels: Number of channels model expects
    
    Returns:
        prediction_map: Full-size prediction map (H, W) or (H, W, C)
        count_map: Pixel-wise count of predictions (for averaging overlaps)
    """
    height, width = img_data.shape[0], img_data.shape[1]
    
    # Prepare img_data to have correct channels
    if img_data.shape[2] != target_channels:
        if img_data.shape[2] > target_channels:
            img_data = img_data[:, :, :target_channels]
        else:
            padding_needed = target_channels - img_data.shape[2]
            padding = np.repeat(img_data[:, :, -1:], padding_needed, axis=2)
            img_data = np.concatenate([img_data, padding], axis=2)
    
    # ✅ FIX: Input should already be preprocessed using preprocess_for_model()
    # Do NOT apply additional normalization here!
    # The img_data passed in should be the output of preprocess_for_model()
    
    # Initialize prediction and count maps
    # Check model output shape to determine prediction map size
    dummy_input = np.zeros((1, window_size, window_size, target_channels), dtype=np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    
    if len(dummy_output.shape) == 4:  # Segmentation output (batch, H, W, C)
        output_channels = dummy_output.shape[-1]
        prediction_map = np.zeros((height, width, output_channels), dtype=np.float32)
    else:  # Classification or other output
        prediction_map = np.zeros((height, width), dtype=np.float32)
    
    count_map = np.zeros((height, width), dtype=np.float32)
    
    # Slide window across image
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract window
            window = img_data[y:y+window_size, x:x+window_size, :]
            
            # Skip if window is not correct size (edge case)
            if window.shape[0] != window_size or window.shape[1] != window_size:
                continue
            
            # Add batch dimension and predict
            window_batch = np.expand_dims(window, axis=0)
            prediction = model.predict(window_batch, verbose=0)
            
            # Handle different output types
            if len(prediction.shape) == 4:  # Segmentation (batch, H, W, C)
                pred_window = prediction[0]  # Remove batch dimension
                prediction_map[y:y+window_size, x:x+window_size, :] += pred_window
            elif len(prediction.shape) == 2:  # Classification (batch, classes)
                # For classification, we assign the predicted class to the center of window
                pred_class = np.argmax(prediction[0])
                center_y = y + window_size // 2
                center_x = x + window_size // 2
                prediction_map[center_y, center_x] += pred_class
            else:
                # Generic case: spread prediction across window
                pred_value = prediction[0]
                if isinstance(pred_value, (list, np.ndarray)):
                    pred_value = pred_value[0] if len(pred_value) > 0 else 0
                prediction_map[y:y+window_size, x:x+window_size] += pred_value
            
            # Update count map
            count_map[y:y+window_size, x:x+window_size] += 1
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
    
    if len(prediction_map.shape) == 3:
        for c in range(prediction_map.shape[2]):
            prediction_map[:, :, c] /= count_map
    else:
        prediction_map /= count_map
    
    return prediction_map, count_map


# ============================================================
# UNIFIED MODEL PREDICTION FUNCTION (NDSI-style)
# ============================================================

def run_model_prediction_UNIFIED():
    """
    🎯 UNIFIED PREDICTION FUNCTION
    
    Similar to NDSI tab's run_calculation():
    - If image is selected from listbox → Single image prediction
    - If NO image selected → Batch prediction for ALL images in folder
    
    This replaces separate single/batch prediction functions with one unified approach.
    """
    global loaded_model, model_folder_path, model_image_files, model_input_image_path, model_output_folder
    
    # Validation checks
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please load a model first!")
        return
    
    if not model_folder_path or not model_image_files:
        messagebox.showerror("Error", "Please select a folder with images first!")
        return
    
    # Check if an image is selected from listbox
    selection = model_image_listbox.curselection() if model_image_listbox else ()
    
    if selection and model_input_image_path:
        # ✅ SINGLE IMAGE MODE - Image is selected
        print("\n" + "="*80)
        print("🎯 SINGLE IMAGE PREDICTION MODE")
        print("="*80)
        run_single_image_prediction()
    else:
        # ✅ BATCH MODE - No image selected, process all images in folder
        print("\n" + "="*80)
        print("📦 BATCH PREDICTION MODE")
        print("="*80)
        
        # Check if output folder is selected
        if not model_output_folder:
            messagebox.showinfo(
                "Select Output Folder",
                "Please select an output folder for batch prediction results.\n\n"
                "Results will be saved there."
            )
            return
        
        run_batch_prediction_UNIFIED()


def run_single_image_prediction():
    """
    Run prediction on a single selected image (with 8-band composite).
    This is the CORRECTED version that uses ALL 8 bands.
    """
    global model_input_image_path, loaded_model, model_prediction_result
    
    if model_input_image_path is None:
        messagebox.showwarning("No Image", "Please select an image first!")
        return
    
    try:
        import tensorflow as tf
    except ImportError:
        messagebox.showerror(
            "TensorFlow Not Found",
            "TensorFlow is not installed!\n\nPlease install it using:\npip install tensorflow"
        )
        return
    
    print("="*80)
    print("🚀 STARTING SINGLE IMAGE PREDICTION WITH 8-BAND COMPOSITE")
    print("="*80)
    
    set_status("⏳ Loading composite image (8 bands)...", "warning")
    root.update()
    
    # 🔧 FIX 1: Load composite with ALL 8 bands
    composite = load_composite_image_8_bands(model_input_image_path)
    
    if composite is None:
        base_name = os.path.splitext(os.path.basename(model_input_image_path))[0]
        dir_path = os.path.dirname(model_input_image_path)
        
        error_msg = (
            "❌ 8-BAND COMPOSITE IMAGE NOT FOUND!\n\n"
            f"Original image: {os.path.basename(model_input_image_path)}\n\n"
            "The model REQUIRES a preprocessed composite containing:\n"
            "  • Bands 1-5: Blue, Green, Red, NIR, SWIR (normalized)\n"
            "  • Bands 6-8: NDSI, NDWI, NDVI (spectral indices)\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 HOW TO FIX THIS:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "STEP 1: Go to the 'NDSI Calculation' tab\n\n"
            "STEP 2: Load your image using 'Select Input Folder'\n\n"
            "STEP 3: Select your image from the list\n\n"
            "STEP 4: Click 'Run All Indices' button\n"
            "   → This will generate the 8-band composite\n"
            "   → Wait for processing to complete\n\n"
            "STEP 5: Return to 'Load Model' tab\n\n"
            "STEP 6: Click 'Run Prediction' again\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Expected composite file location:\n"
            f"  {dir_path}/\n"
            f"  └─ {base_name}_processed_composite.tiff\n\n"
            "⚠️ Without this composite, the model CANNOT make accurate predictions!"
        )
        
        messagebox.showerror("Composite Not Found", error_msg)
        set_status("✗ Composite not found - Please run 'All Indices' first", "error")
        return
    
    height, width, channels = composite.shape
    print(f"\n📊 Composite Info:")
    print(f"   Shape: {composite.shape}")
    print(f"   Dtype: {composite.dtype}")
    print(f"   Value range: [{composite.min():.4f}, {composite.max():.4f}]")
    
    if channels != 8:
        messagebox.showerror(
            "Invalid Composite",
            f"Composite has {channels} bands but expected 8!\n\n"
            "Please regenerate the composite using 'All Indices' calculation."
        )
        return
    
    set_status(f"✓ Loaded {height}x{width} image with {channels} bands", "success")
    root.update()
    
    # Apply preprocessing (same as validation)
    print(f"\n🔄 APPLYING PER-BAND STANDARDIZATION (same as validation):")
    input_data, preprocess_log = preprocess_for_model(composite)
    
    for log_line in preprocess_log.split('\n'):
        if log_line.strip():
            print(f"   {log_line}")
    
    print(f"\n📥 Preprocessed Input Data:")
    print(f"   Shape: {input_data.shape}")
    print(f"   Dtype: {input_data.dtype}")
    
    # Window size validation
    model_expected_size = loaded_model.input_shape[1]
    
    if window_size != model_expected_size:
        messagebox.showerror(
            "Window Size Mismatch!",
            f"❌ ERROR: Window size doesn't match model input!\n\n"
            f"Model expects: {model_expected_size}x{model_expected_size}\n"
            f"Current setting: {window_size}x{window_size}\n\n"
            f"Fix: Reload the model or set window_size = {model_expected_size}"
        )
        set_status("❌ Window size mismatch!", "error")
        return
    
    set_status(f"⏳ Running prediction (window: {window_size}x{window_size}, stride: {stride})...", "warning")
    root.update()
    
    # Run prediction with sliding window
    input_shape = loaded_model.input_shape
    target_height = input_shape[1] if input_shape[1] is not None else 256
    target_width = input_shape[2] if input_shape[2] is not None else 256
    target_channels = input_shape[3]
    
    if height > target_height or width > target_width:
        print(f"\n🔄 Using sliding window prediction...")
        prediction_map, count_map = predict_with_sliding_window(
            loaded_model,
            input_data,
            window_size,
            stride,
            target_channels
        )
        
        pred_mask = prediction_map
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
    else:
        print(f"\n🖼️ Using single prediction (image smaller than model input)...")
        
        # Resize and predict (same logic as before)
        resized_channels = []
        for c in range(8):
            channel = input_data[:, :, c]
            ch_min, ch_max = channel.min(), channel.max()
            if ch_max > ch_min:
                channel_norm = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
            else:
                channel_norm = np.zeros_like(channel, dtype=np.uint8)
            
            pil_ch = PILImage.fromarray(channel_norm, mode='L')
            pil_ch_resized = pil_ch.resize((target_width, target_height), PILImage.LANCZOS)
            resized_ch = np.array(pil_ch_resized).astype(np.float32) / 255.0
            
            if ch_max > ch_min:
                resized_ch = resized_ch * (ch_max - ch_min) + ch_min
            
            resized_channels.append(resized_ch)
        
        img_array = np.stack(resized_channels, axis=2)
        img_batch = np.expand_dims(img_array, axis=0)
        
        predictions = loaded_model.predict(img_batch, verbose=0)
        pred_result = predictions[0]
        pred_mask = pred_result[:, :, 0]
        
        # Resize back
        pred_mask_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
        pred_mask_pil = pred_mask_pil.resize((width, height), PILImage.LANCZOS)
        pred_mask = np.array(pred_mask_pil).astype(np.float32) / 255.0
    
    print(f"\n📈 Prediction Statistics:")
    print(f"   Range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
    print(f"   Mean: {pred_mask.mean():.4f}")
    
    # Apply threshold and calculate stats
    threshold = 0.5
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    
    total_pixels = binary_mask.size
    positive_pixels = np.sum(binary_mask)
    positive_percentage = (positive_pixels / total_pixels) * 100
    
    mean_confidence = np.mean(pred_mask)
    max_confidence = np.max(pred_mask)
    min_confidence = np.min(pred_mask)
    std_confidence = np.std(pred_mask)
    
    # Display results
    result_text = f"""🎯 PREDICTION RESULTS:

📏 Image Size: {height} × {width}
🎯 Threshold: {threshold}

📊 Pixel Classification:
  Total Pixels: {total_pixels:,}
  ✅ Positive (Snow): {positive_pixels:,} ({positive_percentage:.2f}%)
  ❌ Negative (No Snow): {total_pixels - positive_pixels:,} ({100-positive_percentage:.2f}%)

💯 Confidence Statistics:
  Mean: {mean_confidence:.4f}
  Std Dev: {std_confidence:.4f}
  Max: {max_confidence:.4f}
  Min: {min_confidence:.4f}
"""
    
    if model_result_label:
        model_result_label.config(text=result_text, fg=COLORS['success'])
    
    set_status("✓ Prediction completed successfully", "success")
    
    # Save result
    model_prediction_result = {
        'prediction': pred_mask,
        'binary': binary_mask,
        'stats': {
            'positive': int(positive_pixels),
            'negative': int(total_pixels - positive_pixels),
            'mean': float(mean_confidence),
            'std': float(std_confidence),
            'max': float(max_confidence),
            'min': float(min_confidence)
        }
    }
    
    # Visualize
    try:
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        color_mask[binary_mask == 1] = [255, 0, 0]  # Red for positive
        
        display_img = PILImage.fromarray(color_mask)
        
        if model_preview_canvas:
            canvas_width = model_preview_canvas.winfo_width() or 600
            canvas_height = model_preview_canvas.winfo_height() or 200
            
            img_width, img_height = display_img.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
            
            display_img = display_img.resize(
                (int(img_width * scale), int(img_height * scale)),
                PILImage.LANCZOS
            )
            
            photo = ImageTk.PhotoImage(display_img)
            model_preview_canvas.delete("all")
            model_preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor='center'
            )
            model_preview_canvas.image = photo
    except Exception as viz_error:
        print(f"⚠️ Visualization error: {viz_error}")
    
    print("="*80)
    print("✅ SINGLE IMAGE PREDICTION COMPLETE")
    print("="*80)
    
    messagebox.showinfo(
        "Prediction Complete",
        f"✅ Prediction completed!\n\n"
        f"Positive pixels: {positive_percentage:.2f}%\n"
        f"Negative pixels: {100-positive_percentage:.2f}%\n\n"
        f"Mean confidence: {mean_confidence:.4f}"
    )


def run_batch_prediction_UNIFIED():
    """
    Run prediction on ALL images in the folder (batch mode).
    Similar to NDSI tab's batch_process_all_images() function.
    """
    global loaded_model, model_folder_path, model_image_files, model_output_folder
    
    if not model_folder_path or not model_image_files:
        messagebox.showerror("Error", "No folder or images loaded!")
        return
    
    if not model_output_folder:
        messagebox.showinfo("No Output Folder", "Please select an output folder before running batch prediction.")
        return
    
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please load a model first!")
        return
    
    try:
        import tensorflow as tf
        
        # Confirm batch processing
        if not messagebox.askyesno(
            "Confirm Batch Prediction",
            f"Run prediction on {len(model_image_files)} images?\n\n"
            f"Output folder: {model_output_folder}\n\n"
            "This may take some time..."
        ):
            return
        
        set_status(f"🚀 Batch processing {len(model_image_files)} images...", "warning")
        root.update()
        
        processed_count = 0
        skipped_count = 0
        results_data = []
        threshold = 0.5
        
        # Get model info
        model_expected_size = loaded_model.input_shape[1]
        
        if window_size != model_expected_size:
            messagebox.showerror(
                "Window Size Mismatch!",
                f"❌ ERROR: Window size mismatch!\n\n"
                f"Model expects: {model_expected_size}x{model_expected_size}\n"
                f"Current setting: {window_size}x{window_size}\n\n"
                f"Batch prediction ABORTED!\n\n"
                f"Fix: Reload the model or set window_size = {model_expected_size}"
            )
            return
        
        # Process each image
        for idx, img_file in enumerate(model_image_files):
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            set_status(f"⏳ Processing {idx+1}/{len(model_image_files)}: {base_name}", "warning")
            root.update()
            
            try:
                # Check if file is already a composite (8 bands) or needs to load composite
                with rasterio.open(img_file) as src:
                    file_band_count = src.count
                
                if file_band_count == 8:
                    # This IS already a composite file - load it directly
                    print(f"Loading composite directly: {base_name}")
                    with rasterio.open(img_file) as src:
                        composite = np.stack([src.read(i+1) for i in range(8)], axis=-1)
                else:
                    # This is a processed file - need to find composite
                    print(f"Searching for composite for: {base_name}")
                    composite = load_composite_image_8_bands(img_file)
                
                if composite is None:
                    print(f"Skipping {base_name}: No composite found")
                    skipped_count += 1
                    continue
                
                if composite.shape[2] != 8:
                    print(f"Skipping {base_name}: Composite has {composite.shape[2]} bands, expected 8")
                    skipped_count += 1
                    continue
                
                # Preprocess
                input_data, _ = preprocess_for_model(composite)
                
                height, width = composite.shape[0], composite.shape[1]
                
                # Predict using sliding window
                input_shape = loaded_model.input_shape
                target_height = input_shape[1] if input_shape[1] is not None else 256
                target_width = input_shape[2] if input_shape[2] is not None else 256
                
                if height > target_height or width > target_width:
                    pred_mask, _ = predict_with_sliding_window(
                        loaded_model,
                        input_data,
                        window_size,
                        stride,
                        8
                    )
                    if len(pred_mask.shape) == 3:
                        pred_mask = pred_mask[:, :, 0]
                else:
                    # Resize and predict (simplified for batch)
                    resized_channels = []
                    for c in range(8):
                        channel = input_data[:, :, c]
                        ch_min, ch_max = channel.min(), channel.max()
                        if ch_max > ch_min:
                            channel_norm = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
                        else:
                            channel_norm = np.zeros_like(channel, dtype=np.uint8)
                        
                        pil_ch = PILImage.fromarray(channel_norm, mode='L')
                        pil_ch_resized = pil_ch.resize((target_width, target_height), PILImage.LANCZOS)
                        resized_ch = np.array(pil_ch_resized).astype(np.float32) / 255.0
                        
                        if ch_max > ch_min:
                            resized_ch = resized_ch * (ch_max - ch_min) + ch_min
                        
                        resized_channels.append(resized_ch)
                    
                    img_array = np.stack(resized_channels, axis=2)
                    img_batch = np.expand_dims(img_array, axis=0)
                    predictions = loaded_model.predict(img_batch, verbose=0)
                    pred_result = predictions[0]
                    pred_mask = pred_result[:, :, 0]
                    
                    # Resize back
                    pred_mask_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    pred_mask_pil = pred_mask_pil.resize((width, height), PILImage.LANCZOS)
                    pred_mask = np.array(pred_mask_pil).astype(np.float32) / 255.0
                
                # Calculate statistics
                binary_mask = (pred_mask > threshold).astype(np.uint8)
                total_pixels = binary_mask.size
                positive_pixels = np.sum(binary_mask)
                positive_percentage = (positive_pixels / total_pixels) * 100
                
                mean_confidence = float(np.mean(pred_mask))
                max_confidence = float(np.max(pred_mask))
                
                # ✅ Save ONLY probability .tiff
                output_path = os.path.join(model_output_folder, f"{base_name}_prediction_probability.tif")
                saved_file = save_geotiff_probability_only(
                    pred_mask=pred_mask,
                    original_img_path=img_file,
                    output_path=output_path
                )
                
                # Store results (simplified - only probability file)
                result_dict = {
                    'filename': os.path.basename(img_file),
                    'total_pixels': total_pixels,
                    'positive_pixels': int(positive_pixels),
                    'positive_percentage': positive_percentage,
                    'mean_confidence': mean_confidence,
                    'max_confidence': max_confidence,
                    'threshold': threshold,
                    'probability_tif': saved_file
                }
                results_data.append(result_dict)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                skipped_count += 1
                continue
        
        # Save summary CSV
        import csv
        csv_path = os.path.join(model_output_folder, "batch_prediction_results.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'total_pixels', 'positive_pixels', 'positive_percentage',
                         'mean_confidence', 'max_confidence', 'threshold', 'mask_path',
                         'probability_tif', 'colorized_tif', 'preview_png', 'report_txt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results_data:
                writer.writerow(result)
        
        set_status(f"✅ Batch processing complete: {processed_count}/{len(model_image_files)} successful", "success")
        
        summary_msg = f"Batch Prediction Complete!\n\n"
        summary_msg += f"✅ Successfully processed: {processed_count}/{len(model_image_files)}\n"
        if skipped_count > 0:
            summary_msg += f"⚠️ Skipped (no composite): {skipped_count}\n"
        summary_msg += f"\n📊 Results saved to:\n{csv_path}"
        
        messagebox.showinfo("Batch Complete", summary_msg)
        
    except Exception as e:
        messagebox.showerror("Batch Error", f"Batch processing failed:\n{str(e)}")
        set_status("❌ Batch processing failed", "error")
        import traceback
        traceback.print_exc()


# ============================================================
# ORIGINAL PREDICTION FUNCTION (kept for reference)
# ============================================================

def run_model_prediction_CORRECTED():
    """
    🔧 FIXED VERSION: Uses ALL 8 BANDS from composite image
    
    Key fixes:
    1. Loads composite image with all 8 bands
    2. Uses all bands as input (not just first 3) ← MAIN FIX!
    3. Proper normalization based on training data
    4. Better error handling and debugging
    """
    global loaded_model, model_input_image_path, model_prediction_result
    global model_result_label, model_preview_canvas, window_size, stride
    
    # Validate inputs
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please load a model first!")
        return
    
    if model_input_image_path is None:
        messagebox.showwarning("No Image", "Please select an image first!")
        return
    
    try:
        # Import tensorflow
        try:
            import tensorflow as tf
        except ImportError:
            messagebox.showerror(
                "TensorFlow Not Found",
                "TensorFlow is not installed!\\n\\nPlease install it using:\\npip install tensorflow"
            )
            return
        
        print("="*80)
        print("🚀 STARTING PREDICTION WITH 8-BAND COMPOSITE")
        print("="*80)
        
        set_status("⏳ Loading composite image (8 bands)...", "warning")
        root.update()
        
        # 🔧 FIX 1: Load composite with ALL 8 bands
        composite = load_composite_image_8_bands(model_input_image_path)
        
        if composite is None:
            # Try to determine what went wrong
            base_name = os.path.splitext(os.path.basename(model_input_image_path))[0]
            dir_path = os.path.dirname(model_input_image_path)
            
            error_msg = (
                "❌ 8-BAND COMPOSITE IMAGE NOT FOUND!\n\n"
                f"Original image: {os.path.basename(model_input_image_path)}\n\n"
                "The model REQUIRES a preprocessed composite containing:\n"
                "  • Bands 1-5: Blue, Green, Red, NIR, SWIR (normalized)\n"
                "  • Bands 6-8: NDSI, NDWI, NDVI (spectral indices)\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📋 HOW TO FIX THIS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "STEP 1: Go to the 'NDSI Calculation' tab\n\n"
                "STEP 2: Load your image using 'Select Input Folder'\n\n"
                "STEP 3: Select your image from the list\n\n"
                "STEP 4: Click 'Run All Indices' button\n"
                "   → This will generate the 8-band composite\n"
                "   → Wait for processing to complete\n\n"
                "STEP 5: Return to 'Load Model' tab\n\n"
                "STEP 6: Click 'Run Prediction' again\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Expected composite file location:\n"
                f"  {dir_path}/\n"
                f"  └─ {base_name}_processed_composite.tiff\n\n"
                "⚠️ Without this composite, the model CANNOT make accurate predictions!"
            )
            
            messagebox.showerror("Composite Not Found", error_msg)
            set_status("✗ Composite not found - Please run 'All Indices' first", "error")
            return
        
        height, width, channels = composite.shape
        print(f"\\n📊 Composite Info:")
        print(f"   Shape: {composite.shape}")
        print(f"   Dtype: {composite.dtype}")
        print(f"   Value range: [{composite.min():.4f}, {composite.max():.4f}]")
        
        if channels != 8:
            messagebox.showerror(
                "Invalid Composite",
                f"Composite has {channels} bands but expected 8!\\n\\n"
                "Please regenerate the composite using 'All Indices' calculation."
            )
            return
        
        set_status(f"✓ Loaded {height}x{width} image with {channels} bands", "success")
        root.update()
        
        # Get model input shape
        input_shape = loaded_model.input_shape
        print(f"\\n🧠 Model Info:")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {loaded_model.output_shape}")
        
        if len(input_shape) != 4:
            messagebox.showerror("Model Error", f"Unsupported model input shape: {input_shape}")
            return
        
        target_height = input_shape[1] if input_shape[1] is not None else 256
        target_width = input_shape[2] if input_shape[2] is not None else 256
        target_channels = input_shape[3]
        
        print(f"   Expected: {target_height}x{target_width}x{target_channels}")
        
        if target_channels != 8:
            messagebox.showerror(
                "Channel Mismatch",
                f"Model expects {target_channels} channels but composite has 8!\\n\\n"
                f"Model input shape: {input_shape}\\n"
                f"This model was not trained on 8-band composites."
            )
            return
        
        # 🔧 CRITICAL FIX: USE SAME PREPROCESSING AS VALIDATION
        # The validation function uses per-band STANDARDIZATION (z-score normalization)
        # Not min-max [0,1] normalization!
        print(f"\n🔄 APPLYING PER-BAND STANDARDIZATION (same as validation):")
        
        # Use the exact same preprocessing function as validation
        input_data, preprocess_log = preprocess_for_model(composite)
        
        # Print the preprocessing log for debugging
        for log_line in preprocess_log.split('\n'):
            if log_line.strip():
                print(f"   {log_line}")
        
        print(f"\\n📥 Preprocessed Input Data:")
        print(f"   Shape: {input_data.shape}")
        print(f"   Dtype: {input_data.dtype}")
        
        set_status(f"✓ Normalized {height}x{width} image", "success")
        root.update()
        
        # ✅ CRITICAL VALIDATION: Ensure window_size matches model's expected input size
        model_expected_size = loaded_model.input_shape[1]
        
        if window_size != model_expected_size:
            error_msg = (
                f"❌ WINDOW SIZE MISMATCH DETECTED!\n\n"
                f"🚫 This will cause INCORRECT PREDICTIONS!\n\n"
                f"Model expects: {model_expected_size}x{model_expected_size}\n"
                f"Current window_size: {window_size}x{window_size}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"WHY THIS ERROR?\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Your model was trained on {model_expected_size}x{model_expected_size} patches.\n"
                f"If you use a different window size ({window_size}x{window_size}),\n"
                f"the predictions will be WRONG!\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"HOW TO FIX?\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"OPTION 1 (Recommended - Automatic):\n"
                f"  → Reload the model\n"
                f"  → Window size will auto-update to {model_expected_size}\n\n"
                f"OPTION 2 (Manual):\n"
                f"  → Go to 'Prediction Settings' in this tab\n"
                f"  → Set Window Size = {model_expected_size}\n"
                f"  → Click 'Apply Settings'\n"
            )
            
            messagebox.showerror("Window Size Mismatch!", error_msg)
            set_status("❌ Window size doesn't match model input size!", "error")
            return
        
        print(f"✅ Window size validation passed: {window_size} == {model_expected_size}")
        
        set_status(f"⏳ Running prediction (window: {window_size}x{window_size}, stride: {stride})...", "warning")
        root.update()
        
        # 🔧 FIX 3: Run prediction with normalized 8 bands
        if height > target_height or width > target_width:
            print(f"\\n🔄 Using sliding window prediction...")
            print(f"   Window size: {window_size}x{window_size}")
            print(f"   Stride: {stride}")
            print(f"   Overlap: {window_size - stride} pixels")
            
            # Use sliding window
            prediction_map, count_map = predict_with_sliding_window(
                loaded_model,
                input_data,
                window_size,
                stride,
                target_channels
            )
            
            pred_mask = prediction_map
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[:, :, 0]
            
            print(f"✓ Sliding window complete")
            
        else:
            print(f"\\n🖼️ Using single prediction (image smaller than model input)...")
            
            # Resize each channel separately to target size
            resized_channels = []
            for c in range(8):
                channel = input_data[:, :, c]
                
                # Normalize to 0-255 for PIL
                ch_min, ch_max = channel.min(), channel.max()
                if ch_max > ch_min:
                    channel_norm = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
                else:
                    channel_norm = np.zeros_like(channel, dtype=np.uint8)
                
                # Resize using PIL
                pil_ch = PILImage.fromarray(channel_norm, mode='L')
                pil_ch_resized = pil_ch.resize((target_width, target_height), PILImage.LANCZOS)
                
                # Convert back to 0-1 range
                resized_ch = np.array(pil_ch_resized).astype(np.float32) / 255.0
                
                # Rescale back to original range
                if ch_max > ch_min:
                    resized_ch = resized_ch * (ch_max - ch_min) + ch_min
                
                resized_channels.append(resized_ch)
            
            img_array = np.stack(resized_channels, axis=2)
            print(f"   Resized to: {img_array.shape}")
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            print(f"   Batch shape: {img_batch.shape}")
            
            # Predict
            print("   Running prediction...")
            predictions = loaded_model.predict(img_batch, verbose=0)
            print(f"   Prediction shape: {predictions.shape}")
            
            pred_result = predictions[0]
            pred_mask = pred_result[:, :, 0]
            
            # Resize back to original size
            pred_mask_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
            pred_mask_pil = pred_mask_pil.resize((width, height), PILImage.LANCZOS)
            pred_mask = np.array(pred_mask_pil).astype(np.float32) / 255.0
            
            print(f"   Final prediction shape: {pred_mask.shape}")
        
        print(f"\\n📈 Prediction Statistics:")
        print(f"   Range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
        print(f"   Mean: {pred_mask.mean():.4f}")
        print(f"   Std: {pred_mask.std():.4f}")
        
        # Apply threshold
        threshold = 0.5
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        
        # Calculate statistics
        total_pixels = binary_mask.size
        positive_pixels = np.sum(binary_mask)
        negative_pixels = total_pixels - positive_pixels
        
        positive_percentage = (positive_pixels / total_pixels) * 100
        negative_percentage = (negative_pixels / total_pixels) * 100
        
        mean_confidence = np.mean(pred_mask)
        max_confidence = np.max(pred_mask)
        min_confidence = np.min(pred_mask)
        std_confidence = np.std(pred_mask)
        
        print(f"\\n🎯 Classification Results:")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Positive (snow): {positive_pixels:,} ({positive_percentage:.2f}%)")
        print(f"   Negative (no snow): {negative_pixels:,} ({negative_percentage:.2f}%)")
        print(f"   Confidence - Mean: {mean_confidence:.4f}, Std: {std_confidence:.4f}")
        print(f"   Confidence - Min: {min_confidence:.4f}, Max: {max_confidence:.4f}")
        
        # Display results
        result_text = f"""🎯 PREDICTION RESULTS:

📏 Image Size: {height} × {width}
🎯 Threshold: {threshold}

📊 Pixel Classification:
  Total Pixels: {total_pixels:,}
  ✅ Positive (Snow): {positive_pixels:,} ({positive_percentage:.2f}%)
  ❌ Negative (No Snow): {negative_pixels:,} ({negative_percentage:.2f}%)

💯 Confidence Statistics:
  Mean: {mean_confidence:.4f}
  Std Dev: {std_confidence:.4f}
  Max: {max_confidence:.4f}
  Min: {min_confidence:.4f}
"""
        
        if model_result_label:
            model_result_label.config(text=result_text, fg=COLORS['success'])
        
        set_status("✓ Prediction completed successfully", "success")
        
        # Save result
        model_prediction_result = {
            'prediction': pred_mask,
            'binary': binary_mask,
            'stats': {
                'positive': int(positive_pixels),
                'negative': int(negative_pixels),
                'mean': float(mean_confidence),
                'std': float(std_confidence),
                'max': float(max_confidence),
                'min': float(min_confidence)
            }
        }
        
        # Visualize on canvas
        try:
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            color_mask[binary_mask == 1] = [255, 0, 0]  # Red for positive
            
            display_img = PILImage.fromarray(color_mask)
            
            if model_preview_canvas:
                canvas_width = model_preview_canvas.winfo_width() or 600
                canvas_height = model_preview_canvas.winfo_height() or 200
                
                img_width, img_height = display_img.size
                scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
                
                display_img = display_img.resize(
                    (int(img_width * scale), int(img_height * scale)),
                    PILImage.LANCZOS
                )
                
                photo = ImageTk.PhotoImage(display_img)
                model_preview_canvas.delete("all")
                model_preview_canvas.create_image(
                    canvas_width // 2, canvas_height // 2,
                    image=photo, anchor='center'
                )
                model_preview_canvas.image = photo
        except Exception as viz_error:
            print(f"⚠️ Visualization error: {viz_error}")
        
        print("="*80)
        print("✅ PREDICTION COMPLETE")
        print("="*80)
        
        messagebox.showinfo(
            "Prediction Complete",
            f"✅ Prediction completed!\\n\\n"
            f"Positive pixels: {positive_percentage:.2f}%\\n"
            f"Negative pixels: {negative_percentage:.2f}%\\n\\n"
            f"Mean confidence: {mean_confidence:.4f}"
        )
        
    except Exception as e:
        print("="*80)
        print("❌ PREDICTION ERROR")
        print("="*80)
        messagebox.showerror("Prediction Error", f"Failed to run prediction:\\n{str(e)}")
        set_status(f"✗ Error: {str(e)}", "error")
        import traceback
        traceback.print_exc()


def select_batch_folder():
    """Select folder containing images for batch prediction"""
    global model_batch_folder
    
    path = filedialog.askdirectory(title="Select Folder with Images for Batch Prediction")
    if not path:
        return
    
    model_batch_folder = path
    
    # Count image files
    image_files = [f for f in os.listdir(path) 
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        messagebox.showwarning("No Images", f"No image files found in:\n{path}")
        return
    
    set_status(f"📁 Batch folder selected: {len(image_files)} images found", "info")
    messagebox.showinfo("Folder Selected", f"Found {len(image_files)} images for batch processing")


def select_model_output_folder():
    """Select output folder for batch prediction results"""
    global model_output_folder
    
    path = filedialog.askdirectory(title="Select Output Folder for Predictions")
    if not path:
        return
    
    model_output_folder = path
    set_status(f"💾 Output folder selected: {path}", "info")


def run_batch_prediction_CORRECTED():
    """
    CORRECTED batch prediction for SEGMENTATION models
    """
    global loaded_model, model_batch_folder, model_output_folder
    
    # Validate inputs
    if loaded_model is None:
        messagebox.showwarning("No Model", "Please upload a model first!")
        return
    
    if model_batch_folder is None:
        messagebox.showwarning("No Folder", "Please select batch input folder first!")
        return
    
    if model_output_folder is None:
        messagebox.showwarning("No Output Folder", "Please select output folder first!")
        return
    
    try:
        import tensorflow as tf
        
        # Get all image files
        image_files = [f for f in os.listdir(model_batch_folder) 
                       if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showwarning("No Images", "No image files found in selected folder!")
            return
        
        # Confirm
        if not messagebox.askyesno("Confirm Batch Processing", 
                                   f"Run segmentation on {len(image_files)} images?"):
            return
        
        # Get model input shape
        input_shape = loaded_model.input_shape
        
        if len(input_shape) == 4:
            target_height = input_shape[1]
            target_width = input_shape[2]
            target_channels = input_shape[3]
        else:
            messagebox.showerror("Model Error", "Unsupported model input shape!")
            return
        
        # Process each image
        total = len(image_files)
        success_count = 0
        failed_files = []
        results_data = []
        
        threshold = 0.5  # Fixed threshold, make adjustable if needed
        
        # ✅ CRITICAL VALIDATION: Check window_size matches model input
        global window_size, stride
        model_expected_size = loaded_model.input_shape[1]
        
        if window_size != model_expected_size:
            messagebox.showerror(
                "Window Size Mismatch!",
                f"❌ ERROR: Window size mismatch!\n\n"
                f"Model expects: {model_expected_size}x{model_expected_size}\n"
                f"Current window_size: {window_size}x{window_size}\n\n"
                f"Batch prediction ABORTED to prevent incorrect results!\n\n"
                f"Fix: Reload the model or manually set window_size = {model_expected_size}"
            )
            set_status("❌ Batch prediction aborted - window size mismatch", "error")
            return
        
        print(f"✅ Batch prediction validation passed: window_size={window_size}")
        
        for i, filename in enumerate(image_files, 1):
            img_path = os.path.join(model_batch_folder, filename)
            
            set_status(f"⏳ Processing {i}/{total}: {filename}...", "warning")
            root.update()
            
            try:
                # Load and preprocess image (simplified for batch)
                try:
                    with rasterio.open(img_path) as src:
                        if src.count >= target_channels:
                            img_data = np.stack([src.read(j+1) for j in range(target_channels)], axis=-1)
                        else:
                            bands = [src.read(j+1) for j in range(src.count)]
                            while len(bands) < target_channels:
                                bands.append(bands[-1])
                            img_data = np.stack(bands[:target_channels], axis=-1)
                except:
                    pil_img = PILImage.open(img_path).convert('RGB')
                    img_data = np.array(pil_img)
                    if img_data.shape[2] < target_channels:
                        padding = np.repeat(img_data[:, :, -1:], target_channels - img_data.shape[2], axis=2)
                        img_data = np.concatenate([img_data, padding], axis=2)
                
                if img_data.ndim == 2:
                    img_data = np.expand_dims(img_data, axis=-1)
                
                # Get original dimensions
                orig_height, orig_width = img_data.shape[0], img_data.shape[1]
                
                # ==========================================
                # USE SLIDING WINDOW FOR LARGE IMAGES
                # ==========================================
                
                if orig_height > target_height or orig_width > target_width:
                    # Use sliding window prediction
                    pred_mask, count_map = predict_with_sliding_window(
                        loaded_model,
                        img_data,
                        window_size,
                        stride,
                        target_channels
                    )
                    
                    if len(pred_mask.shape) == 3:
                        pred_mask = pred_mask[:, :, 0]
                    
                else:
                    # Image is small - resize and predict normally
                    # Resize using PIL for channels <= 4, scipy for >4
                    if target_channels <= 3:
                        if img_data.dtype != np.uint8:
                            img_min, img_max = img_data.min(), img_data.max()
                            if img_max > img_min:
                                img_data = ((img_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                            else:
                                img_data = np.zeros_like(img_data, dtype=np.uint8)
                        pil_img = PILImage.fromarray(img_data.astype(np.uint8))
                        pil_img = pil_img.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                        img_array = np.array(pil_img)
                    else:
                        # Resize each channel
                        resized_channels = []
                        for c in range(target_channels):
                            ch = img_data[:, :, c] if img_data.shape[2] > c else img_data[:, :, 0]
                            if ch.dtype != np.uint8:
                                ch_min, ch_max = ch.min(), ch.max()
                                if ch_max > ch_min:
                                    ch = ((ch - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
                                else:
                                    ch = np.zeros_like(ch, dtype=np.uint8)
                            pil_ch = PILImage.fromarray(ch.astype(np.uint8), mode='L')
                            pil_ch_resized = pil_ch.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                            resized_channels.append(np.array(pil_ch_resized))
                        img_array = np.stack(resized_channels, axis=2)
                    
                    if img_array.ndim == 2:
                        img_array = np.expand_dims(img_array, axis=-1)
                    
                    # Normalize
                    img_array = img_array.astype(np.float32) / 255.0
                    
                    # Add batch dimension
                    img_batch = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    predictions = loaded_model.predict(img_batch, verbose=0)
                    pred_result = predictions[0]
                    pred_mask = pred_result[:, :, 0]
                    
                    # Resize prediction back to original size
                    pred_mask_pil = PILImage.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    pred_mask_pil = pred_mask_pil.resize((orig_width, orig_height), PILImage.Resampling.LANCZOS)
                    pred_mask = np.array(pred_mask_pil).astype(np.float32) / 255.0
                binary_mask = (pred_mask > threshold).astype(np.uint8)
                
                # Calculate statistics
                total_pixels = binary_mask.size
                positive_pixels = np.sum(binary_mask)
                positive_percentage = (positive_pixels / total_pixels) * 100
                
                mean_confidence = float(np.mean(pred_mask))
                max_confidence = float(np.max(pred_mask))
                
                # Store results
                result_dict = {
                    'filename': filename,
                    'total_pixels': total_pixels,
                    'positive_pixels': int(positive_pixels),
                    'positive_percentage': positive_percentage,
                    'mean_confidence': mean_confidence,
                    'max_confidence': max_confidence,
                    'threshold': threshold
                }
                
                # ✅ Save ONLY probability .tiff
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(model_output_folder, f"{base_name}_segmentation_probability.tif")
                saved_file = save_geotiff_probability_only(
                    pred_mask=pred_mask,
                    original_img_path=img_path,
                    output_path=output_path
                )
                
                result_dict['probability_tif'] = saved_file
                results_data.append(result_dict)
                success_count += 1
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                failed_files.append(filename)
                continue
        
        # Save results summary to CSV
        import csv
        csv_path = os.path.join(model_output_folder, "batch_segmentation_results.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'total_pixels', 'positive_pixels', 'positive_percentage', 
                         'mean_confidence', 'max_confidence', 'threshold', 'probability_tif']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results_data:
                writer.writerow(result)
        
        set_status(f"✅ Batch processing complete: {success_count}/{total} successful", "success")
        
        # Show summary
        summary_msg = f"Batch Segmentation Complete!\\n\\n"
        summary_msg += f"✅ Successfully processed: {success_count}/{total}\\n"
        if failed_files:
            summary_msg += f"❌ Failed: {len(failed_files)}\\n"
            summary_msg += f"\\nFailed files:\\n" + "\\n".join(failed_files[:5])
            if len(failed_files) > 5:
                summary_msg += f"\\n... and {len(failed_files)-5} more"
        summary_msg += f"\\n\\n📊 Results saved to:\\n{csv_path}"
        
        messagebox.showinfo("Batch Complete", summary_msg)
        
    except Exception as e:
        messagebox.showerror("Batch Error", f"Batch processing failed:\\n{str(e)}")
        set_status("❌ Batch processing failed", "error")
        import traceback
        traceback.print_exc()
        





# ---------------- SELECT OUTPUT FOLDER ----------------
def select_output_folder_ndsi():
    """Select output folder for NDSI CALCULATION tab (LEFT COLUMN)"""
    global output_folder_path
    
    print("🔍 NDSI Tab: Opening output folder dialog...")
    
    selected_folder = filedialog.askdirectory(
        title="Select Output Folder for NDSI Results",
        mustexist=False
    )
    
    print(f"🔍 NDSI Tab: Selected folder = '{selected_folder}'")
    
    if selected_folder and selected_folder.strip():
        output_folder_path = selected_folder
        print(f"✅ NDSI Tab: Output folder SET to: {output_folder_path}")
        
        # Update labels for NDSI tab
        try:
            output_folder_display.config(
                text=os.path.basename(output_folder_path),
                fg=COLORS['success']
            )
            print(f"✅ NDSI Tab: Updated output_folder_display label")
        except Exception as e:
            print(f"⚠️  NDSI Tab: Could not update label: {e}")
        
        set_status(f"✓ NDSI Output folder: {os.path.basename(output_folder_path)}", "success")
    else:
        print("❌ NDSI Tab: No folder selected")
        set_status("✗ NDSI output folder selection cancelled", "warning")


def select_output_folder_model():
    """Select output folder for MODEL PREDICTION tab (RIGHT COLUMN)"""
    global model_output_folder, model_output_folder_label, model_output_folder_info
    
    print("🔍 DEBUG: Opening output folder dialog...")
    
    selected_folder = filedialog.askdirectory(
        title="Select Output Folder for Predictions",
        mustexist=False
    )
    
    print(f"🔍 DEBUG: Selected folder = '{selected_folder}'")
    
    if selected_folder and selected_folder.strip():
        model_output_folder = selected_folder
        print(f"✅ DEBUG: Output folder SET to: {model_output_folder}")
        
        # Update labels safely
        try:
            model_output_folder_label.config(
                text=os.path.basename(model_output_folder),
                fg=COLORS['success']
            )
            print(f"✅ DEBUG: Updated model_output_folder_label")
        except Exception as e:
            print(f"⚠️  DEBUG: Could not update model_output_folder_label: {e}")
        
        try:
            model_output_folder_info.config(
                text=f"📁 {model_output_folder}",
                fg=COLORS['text_muted']
            )
            print(f"✅ DEBUG: Updated model_output_folder_info")
        except Exception as e:
            print(f"⚠️  DEBUG: Could not update model_output_folder_info: {e}")
        
        set_status(f"✓ Output folder selected: {os.path.basename(model_output_folder)}", "success")
    else:
        print("❌ DEBUG: No folder selected (user cancelled or empty path)")
        # User cancelled
        try:
            model_output_folder_label.config(
                text="Not selected",
                fg=COLORS['text_muted']
            )
        except:
            pass
        
        try:
            model_output_folder_info.config(text="")
        except:
            pass



# ---------------- UPLOAD KERAS MODEL ----------------
# ---------------- SHOW MODEL DETAILS ----------------
def show_model_details():
    """Show model details in a popup window"""
    global _stored_model_details
    
    if not loaded_model:
        messagebox.showinfo("No Model", "Please load a model first!")
        return
    
    if '_stored_model_details' not in globals() or not _stored_model_details:
        messagebox.showinfo("No Details", "No model details available!")
        return
    
    # Create new window
    details_window = tk.Toplevel(root)
    details_window.title("Model Details")
    details_window.geometry("900x700")
    details_window.configure(bg=COLORS['background'])
    
    # Title
    title_label = tk.Label(
        details_window,
        text="🤖  Model Details",
        font=("Segoe UI", 16, "bold"),
        bg=COLORS['background'],
        fg=COLORS['text']
    )
    title_label.pack(pady=20)
    
    # Scrolled text frame
    details_frame = tk.Frame(details_window, bg=COLORS['card_bg'])
    details_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
    
    # Scrolled text widget
    from tkinter import scrolledtext
    details_text = scrolledtext.ScrolledText(
        details_frame,
        font=("Consolas", 10) if IS_WINDOWS else ("SF Mono", 10) if IS_MAC else ("DejaVu Sans Mono", 10),
        bg=COLORS['details_bg'],
        fg=COLORS['text'],
        wrap='word',
        relief='flat',
        bd=0,
        padx=15,
        pady=15
    )
    details_text.pack(fill='both', expand=True)
    
    # Insert model details
    details_text.insert('1.0', _stored_model_details)
    details_text.config(state='disabled')  # Make read-only
    
    # Close button
    close_btn = tk.Button(
        details_window,
        text="Close",
        font=("Segoe UI", 11, "bold"),
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=10,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=details_window.destroy
    )
    close_btn.pack(pady=(0, 20), padx=20, fill='x')


# ---------------- UPLOAD MODEL ----------------
def upload_model():
    """Upload and load Keras model with support for custom loss functions"""
    global loaded_model, model_path

    model_path = filedialog.askopenfilename(
        title="Select Keras Model File",
        filetypes=[
            ("Keras Model Files", "*.keras *.h5"),
            ("Keras Files", "*.keras"),
            ("HDF5 Files", "*.h5"),
            ("All Files", "*.*")
        ]
    )

    if not model_path:   # guard clause
        return

    _build_model_tab()   # ensure widgets exist before we write to them

    try:
        # Try to import tensorflow/keras
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            messagebox.showerror(
                "TensorFlow Not Found",
                "TensorFlow is not installed!\n\n"
                "Please install it using:\n"
                "pip install tensorflow"
            )
            return

        set_status(f"⏳ Loading model: {os.path.basename(model_path)}...", "warning")
        root.update()

        # 🔧 FIX: Load model WITHOUT compiling first, then compile
        try:
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False
            )
            
            # NOW compile with custom objects
            loaded_model.compile(
                optimizer='adam',
                loss=dice_loss,
                metrics=[dice_coef]
            )
            
        except Exception as e:
            messagebox.showerror(
                "Model Loading Error",
                f"Failed to load model:\n{str(e)}"
            )
            return
        
        # ✅ AUTO-UPDATE window_size based on model input shape
        global window_size, stride, window_size_var, stride_var
        input_shape = loaded_model.input_shape
        detected_size = input_shape[1]  # Get patch size from model (128 or 256)
        
        window_size = detected_size
        stride = detected_size // 2  # 50% overlap by default
        
        # ✅ Also update the GUI entry fields if they exist
        try:
            if window_size_var is not None:
                window_size_var.set(detected_size)
            if stride_var is not None:
                stride_var.set(stride)
        except Exception as e:
            print(f"[WARNING] Could not update GUI fields: {e}")
        
        print(f"[INFO] Auto-detected model patch size: {detected_size}x{detected_size}")
        print(f"[INFO] Updated window_size={window_size}, stride={stride}")
        if window_size_var is not None:
            print(f"[INFO] GUI fields updated automatically")

        # Update UI
        model_filename_display.config(
            text=os.path.basename(model_path),
            fg=COLORS['success']
        )

        # Get model details
        model_info = []
        model_info.append(f"✅ Model loaded successfully!\n")
        model_info.append(f"📁 File: {os.path.basename(model_path)}\n")
        model_info.append(f"📊 Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB\n")
        model_info.append(f"🎯 Detected Patch Size: {window_size}x{window_size}\n")
        model_info.append(f"🔄 Sliding Window: {window_size}x{window_size} (stride={stride})\n\n")

        model_info.append("=" * 50 + "\n")
        model_info.append("MODEL ARCHITECTURE:\n")
        model_info.append("=" * 50 + "\n\n")

        # Get model summary
        summary_lines = []
        loaded_model.summary(print_fn=lambda x: summary_lines.append(x))
        model_info.append("\n".join(summary_lines))

        model_info.append("\n\n" + "=" * 50 + "\n")
        model_info.append(f"Total Parameters: {loaded_model.count_params():,}\n")
        model_info.append("=" * 50 + "\n")

        # Store model details in global variable for popup window
        global _stored_model_details
        _stored_model_details = ''.join(model_info)

        set_status(
            f"✅ Model loaded: {os.path.basename(model_path)} ({loaded_model.count_params():,} parameters)",
            "success"
        )

        messagebox.showinfo(
            "Model Loaded",
            f"Model loaded successfully!\n\n"
            f"File: {os.path.basename(model_path)}\n"
            f"Parameters: {loaded_model.count_params():,}\n"
            f"Layers: {len(loaded_model.layers)}\n\n"
            f"🎯 Detected Input Size: {window_size}x{window_size}\n"
            f"🔄 Sliding Window: {window_size}x{window_size} (stride={stride})"
        )

    except Exception as e:
        messagebox.showerror(
            "Model Load Error",
            f"Failed to load model:\n\n{str(e)}"
        )
        set_status("❌ Failed to load model", "error")
        print(f"Model load error: {e}")
        import traceback
        traceback.print_exc()


# ---------------- DISPLAY IMAGE PREVIEW ----------------
def display_image_preview(img_path):
    """Display image preview in the right panel"""
    try:
        from PIL import Image, ImageTk
        import matplotlib.pyplot as plt
        import io

        # Read the image using rasterio
        with rasterio.open(img_path) as src:
            # Read first band for preview
            band_data = src.read(1)

            # Normalize for display
            band_min = np.nanmin(band_data)
            band_max = np.nanmax(band_data)

            if band_max > band_min:
                normalized = (band_data - band_min) / (band_max - band_min)
            else:
                normalized = band_data

            # Convert to 0-255 range
            preview_data = (normalized * 255).astype(np.uint8)

            # Create PIL Image
            pil_img = Image.fromarray(preview_data, mode='L')

            # Get canvas size
            canvas_width = preview_canvas.winfo_width()
            canvas_height = preview_canvas.winfo_height()

            # If canvas not rendered yet, use default size
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 500

            # Calculate scaling to fit canvas
            img_width, img_height = pil_img.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # Resize image - handle PIL version compatibility
            try:
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)

            # Clear canvas
            preview_canvas.delete("all")

            # Display image centered
            preview_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor='center'
            )

            # Keep reference to prevent garbage collection
            preview_canvas.image = photo

            # Hide the "no image" label
            preview_label.place_forget()

            # Update info label
            image_info_label.config(
                text=f"📐 {img_width} x {img_height} px  |  Band 1 preview",
                fg=COLORS['text_muted']
            )

    except Exception as e:
        print(f"Preview error: {e}")
        preview_canvas.delete("all")
        preview_label.config(
            text=f"Preview not available\n\n{str(e)[:50]}",
            fg=COLORS['error']
        )
        preview_label.place(relx=0.5, rely=0.5, anchor='center')


# ---------------- UPLOAD FOLDER ----------------
def upload_folder():
    global folder_path, image_files, current_image_index, image_path, result_data

    folder_path = filedialog.askdirectory(title="Select Folder Containing GeoTIFF Images")

    if not folder_path:   # 1️⃣2️⃣ guard
        return

    # Find all TIFF files in the folder
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.tif', '.tiff')):
            image_files.append(os.path.join(folder_path, file))

    image_files.sort()  # Sort alphabetically

    if not image_files:
        messagebox.showwarning("No Images Found", "No GeoTIFF files found in the selected folder.")
        folder_path = None
        return

    current_image_index = 0
    result_data = None

    # Update UI
    folder_label.config(
        text=f"{os.path.basename(folder_path)}",
        fg=COLORS['success']
    )

    folder_info_label.config(
        text=f"{len(image_files)} GeoTIFF images found",
        fg=COLORS['text_muted']
    )

    # Show and populate image listbox
    listbox_frame.pack(fill='both', expand=True, pady=(10, 0))
    image_listbox.delete(0, tk.END)
    for img_file in image_files:
        image_listbox.insert(tk.END, os.path.basename(img_file))

    # Select first image
    if image_files:
        image_listbox.selection_set(0)
        load_selected_image()

    set_status(f"✓ Folder loaded with {len(image_files)} images – Select an image to begin", "success")


# ---------------- LOAD SELECTED IMAGE ----------------
def load_selected_image(event=None):
    global image_path, current_image_index, result_data

    selection = image_listbox.curselection()

    if not selection:   # 1️⃣2️⃣ guard
        return

    current_image_index = selection[0]
    image_path = image_files[current_image_index]
    result_data = None

    # 1️⃣1️⃣ invalidate band cache for previous images (keep current if cached)
    # (cache persists; we only clear entries for files that no longer exist)

    # 7️⃣ disable run until we confirm bands
    _update_run_button()   # will re-enable if valid

    try:
        with rasterio.open(image_path) as src:
            band_count = src.count
            width = src.width
            height = src.height
            dtype = str(src.dtypes[0])
            crs = str(src.crs) if src.crs else "Not specified"
            descriptions = src.descriptions

            # Display band information
            band_info_text = f"Bands: {band_count}\n"
            if descriptions:
                band_info_text += "Band names: " + ", ".join([desc if desc else f"Band{i+1}" for i, desc in enumerate(descriptions[:5])])
                if band_count > 5:
                    band_info_text += "..."

        file_name_label.config(
            text=f"Current: {os.path.basename(image_path)}",
            fg=COLORS['success']
        )

        file_info_label.config(
            text=f"{band_count} bands | {width}x{height} pixels",
            fg=COLORS['text_muted']
        )

        if band_count < 2:
            set_status("❌ Error: Image must contain at least 2 bands for index calculation", "error")
            # 7️⃣ force disabled
            if _btn_run:
                _btn_run.config(state='disabled',
                                bg=COLORS['primary_disabled'],
                                activebackground=COLORS['primary_disabled'])
        else:
            set_status(f"✓ Image {current_image_index + 1}/{len(image_files)} loaded – Ready to calculate", "success")
            _update_run_button()   # 7️⃣
            display_image_preview(image_path)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        set_status("❌ Error loading image", "error")


# ---------------- BATCH PROCESS SINGLE INDEX FOR ALL IMAGES ----------------
def batch_process_single_index(calc_type):
    """
    Process ALL images in folder with a single index type (NDSI, NDWI, or NDVI).
    Creates one TIFF file per image.
    """
    global profile, output_folder_path

    if not folder_path or not image_files:
        messagebox.showerror("Error", "No folder or images loaded!")
        return

    if not output_folder_path:
        messagebox.showinfo("No Output Folder", "Please select an output folder before running calculation.")
        return

    try:
        set_status(f"🚀 Batch processing {len(image_files)} images with {calc_type}...", "warning")
        root.update()

        processed_count = 0
        skipped_count = 0
        generated_files = []

        # Process each image separately
        for idx, img_file in enumerate(image_files):
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            set_status(f"⏳ Processing {calc_type} for image {idx+1}/{len(image_files)}: {base_name}", "warning")
            root.update()

            try:
                with rasterio.open(img_file) as src:
                    if idx == 0:
                        profile = src.profile

                    # Detect required bands for this index type
                    band_result = get_required_bands(src, calc_type)

                    if not band_result['success']:
                        missing_info = f"Missing bands: {', '.join(band_result['missing'])}"
                        print(f"Skipping {base_name}: {missing_info}")
                        print(f"  Image has {src.count} bands: {src.descriptions}")
                        set_status(f"⚠️ Skipping {base_name} - {missing_info}", "warning")
                        root.update()
                        skipped_count += 1
                        continue

                    # Extract band data
                    bands = band_result['bands']

                    # Calculate index based on type
                    if calc_type == 'NDSI':
                        band1_data = bands['GREEN']['data']
                        band2_data = bands['SWIR']['data']
                    elif calc_type == 'NDWI':
                        band1_data = bands['GREEN']['data']
                        band2_data = bands['NIR']['data']
                    elif calc_type == 'NDVI':
                        band1_data = bands['NIR']['data']
                        band2_data = bands['RED']['data']

                    # Calculate index
                    denom = band1_data + band2_data

                    with np.errstate(divide='ignore', invalid='ignore'):
                        result_data = (band1_data - band2_data) / denom

                    result_data[denom == 0] = np.nan
                    result_data = np.clip(result_data, -1, 1)

                    # FIX: Save profile BEFORE closing dataset
                    temp_profile = src.profile.copy()

                # Save TIFF file for this image
                output_filename = f"{base_name}_{calc_type}_processed.tiff"
                output_path = os.path.join(output_folder_path, output_filename)

                # FIX: Use saved profile (dataset is now closed)
                temp_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

                with rasterio.open(output_path, 'w', **temp_profile) as dst:
                    dst.write(result_data.astype(np.float32), 1)
                    dst.set_band_description(1, f'{calc_type}')

                generated_files.append(output_filename)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                skipped_count += 1
                continue

        if processed_count == 0:
            messagebox.showerror("Error", "No valid images could be processed!")
            set_status("❌ No valid images", "error")
            return

        set_status(f"✅ Batch processing complete! Processed {processed_count} images", "success")

        # Show detailed summary
        summary_text = (
            f"Batch Processing Results:\n\n"
            f"✅ Successfully processed: {processed_count} images with {calc_type}\n"
            f"⚠️ Skipped: {skipped_count} images\n\n"
        )
        
        if skipped_count > 0:
            summary_text += (
                f"Note: {skipped_count} images were skipped because they were missing\n"
                f"required bands for {calc_type} calculation.\n"
                f"Check the console for detailed band information.\n\n"
            )
        
        summary_text += f"Location: {output_folder_path}\n\n"
        summary_text += f"Generated {processed_count} {calc_type} files:\n"
        
        # Show first few filenames
        for i, filename in enumerate(generated_files[:5]):
            summary_text += f"• {filename}\n"
        
        if len(generated_files) > 5:
            summary_text += f"... and {len(generated_files) - 5} more files\n"

        messagebox.showinfo(
            "Batch Processing Complete",
            summary_text
        )

    except Exception as e:
        messagebox.showerror("Batch Processing Error", f"Failed to process images:\n{str(e)}")
        set_status("❌ Batch processing failed", "error")
        import traceback
        traceback.print_exc()


# ---------------- BATCH PROCESS ALL IMAGES IN FOLDER ----------------
def batch_process_all_images():
    """
    Process ALL images in the folder when no specific image is selected.
    Creates separate composite TIFF file for EACH image.
    """
    global profile, output_folder_path

    if not folder_path or not image_files:
        messagebox.showerror("Error", "No folder or images loaded!")
        return

    if not output_folder_path:
        messagebox.showinfo("No Output Folder", "Please select an output folder before running calculation.")
        return

    try:
        set_status(f"🚀 Batch processing {len(image_files)} images...", "warning")
        root.update()

        processed_count = 0
        skipped_count = 0
        generated_files = []

        # Process each image separately
        for idx, img_file in enumerate(image_files):
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            set_status(f"⏳ Processing image {idx+1}/{len(image_files)}: {base_name}", "warning")
            root.update()

            try:
                with rasterio.open(img_file) as src:
                    if idx == 0:
                        profile = src.profile

                    # Detect bands
                    blue_idx  = _cached_detect_band(src, 'BLUE')
                    green_idx = _cached_detect_band(src, 'GREEN')
                    red_idx   = _cached_detect_band(src, 'RED')
                    nir_idx   = _cached_detect_band(src, 'NIR')
                    swir_idx  = _cached_detect_band(src, 'SWIR')

                    # Check if all required bands are detected
                    missing_bands = []
                    if not blue_idx:  missing_bands.append('BLUE')
                    if not green_idx: missing_bands.append('GREEN')
                    if not red_idx:   missing_bands.append('RED')
                    if not nir_idx:   missing_bands.append('NIR')
                    if not swir_idx:  missing_bands.append('SWIR')

                    if missing_bands:
                        print(f"Skipping {base_name}: Missing bands: {', '.join(missing_bands)}")
                        print(f"  Image has {src.count} bands: {src.descriptions}")
                        set_status(f"⚠️ Skipping {base_name} - missing: {', '.join(missing_bands)}", "warning")
                        root.update()
                        skipped_count += 1
                        continue

                    # Read bands
                    blue  = src.read(blue_idx).astype(float)
                    green = src.read(green_idx).astype(float)
                    red   = src.read(red_idx).astype(float)
                    nir   = src.read(nir_idx).astype(float)
                    swir  = src.read(swir_idx).astype(float)

                    # FIX: Save profile BEFORE closing dataset
                    composite_profile = src.profile.copy()

                    # Calculate indices
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndsi = (green - swir) / (green + swir)
                        ndsi = np.where(np.isfinite(ndsi), ndsi, np.nan)

                        ndwi = (green - nir) / (green + nir)
                        ndwi = np.where(np.isfinite(ndwi), ndwi, np.nan)

                        ndvi = (nir - red) / (nir + red)
                        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

                # Save individual composite TIFF for THIS image
                composite_filename = f"{base_name}_processed_composite.tiff"
                composite_tiff_path = os.path.join(output_folder_path, composite_filename)

                # FIX: Use saved profile instead of src.profile (dataset is now closed)
                composite_profile.update(
                    dtype=rasterio.float32,
                    count=8,
                    nodata=np.nan
                )

                with rasterio.open(composite_tiff_path, 'w', **composite_profile) as dst:
                    dst.write((blue / SCALING_FACTOR).astype(np.float32), 1)
                    dst.write((green / SCALING_FACTOR).astype(np.float32), 2)
                    dst.write((red / SCALING_FACTOR).astype(np.float32), 3)
                    dst.write((nir / SCALING_FACTOR).astype(np.float32), 4)
                    dst.write((swir / SCALING_FACTOR).astype(np.float32), 5)
                    dst.write(ndsi.astype(np.float32), 6)
                    dst.write(ndwi.astype(np.float32), 7)
                    dst.write(ndvi.astype(np.float32), 8)

                    dst.set_band_description(1, 'BLUE (B2)')
                    dst.set_band_description(2, 'GREEN (B3)')
                    dst.set_band_description(3, 'RED (B4)')
                    dst.set_band_description(4, 'NIR (B8)')
                    dst.set_band_description(5, 'SWIR (B11)')
                    dst.set_band_description(6, 'NDSI (Snow Index)')
                    dst.set_band_description(7, 'NDWI (Water Index)')
                    dst.set_band_description(8, 'NDVI (Vegetation Index)')

                generated_files.append(composite_filename)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                skipped_count += 1
                continue

        if processed_count == 0:
            messagebox.showerror("Error", "No valid images could be processed!")
            set_status("❌ No valid images", "error")
            return

        set_status(f"✅ Batch processing complete! Processed {processed_count} images", "success")

        # Show detailed summary
        summary_text = (
            f"Successfully processed {processed_count} images!\n"
            f"Skipped: {skipped_count} images\n\n"
            f"Location: {output_folder_path}\n\n"
            f"Generated {processed_count} composite files (8-band each):\n"
        )
        
        # Show first few filenames
        for i, filename in enumerate(generated_files[:5]):
            summary_text += f"• {filename}\n"
        
        if len(generated_files) > 5:
            summary_text += f"... and {len(generated_files) - 5} more files\n"

        messagebox.showinfo(
            "Batch Processing Complete",
            summary_text
        )

    except Exception as e:
        messagebox.showerror("Batch Processing Error", f"Failed to process images:\n{str(e)}")
        set_status("❌ Batch processing failed", "error")
        import traceback
        traceback.print_exc()


# ---------------- CALCULATE ALL INDICES ----------------
def calculate_all_indices():
    """
    Calculate NDSI, NDWI, and NDVI in one go from OPTICAL image,
    and create an 8-band composite TIFF with:
    Band 1: BLUE (B2)
    Band 2: GREEN (B3)
    Band 3: RED (B4)
    Band 4: NIR (B8)
    Band 5: SWIR (B11)
    Band 6: NDSI
    Band 7: NDWI
    Band 8: NDVI
    """
    global profile, green, swir, output_folder_path

    # 1️⃣2️⃣ guard clauses
    if not image_path:
        messagebox.showerror("Error", "Please load an image first!")
        return
    if not output_folder_path:
        messagebox.showinfo("No Output Folder", "Please select an output folder before running calculation.")
        return

    try:
        set_status("🔍 Detecting all required bands...", "warning")
        root.update()

        with rasterio.open(image_path) as src:
            profile = src.profile

            # ============ DETECT ALL 5 OPTICAL BANDS (1️⃣1️⃣ cached) ============
            blue_idx  = _cached_detect_band(src, 'BLUE')
            green_idx = _cached_detect_band(src, 'GREEN')
            red_idx   = _cached_detect_band(src, 'RED')
            nir_idx   = _cached_detect_band(src, 'NIR')
            swir_idx  = _cached_detect_band(src, 'SWIR')

            # Check if all required bands are detected
            missing_bands = []
            if not blue_idx:  missing_bands.append('BLUE (B2)')
            if not green_idx: missing_bands.append('GREEN (B3)')
            if not red_idx:   missing_bands.append('RED (B4)')
            if not nir_idx:   missing_bands.append('NIR (B8)')
            if not swir_idx:  missing_bands.append('SWIR (B11)')

            if missing_bands:
                messagebox.showerror(
                    "Missing Bands",
                    f"Could not detect the following required bands:\n\n" +
                    "\n".join([f"• {band}" for band in missing_bands]) +
                    f"\n\nImage has {src.count} bands with descriptions:\n{src.descriptions}\n\n"
                    "Please ensure your image is a Sentinel-2 optical image with bands B2, B3, B4, B8, B11"
                )
                set_status("❌ Missing required bands", "error")
                return

            set_status("✅ All bands detected! Reading data...", "success")
            root.update()

            # ============ READ ALL 5 OPTICAL BANDS ============
            blue  = src.read(blue_idx).astype(float)
            green = src.read(green_idx).astype(float)
            red   = src.read(red_idx).astype(float)
            nir   = src.read(nir_idx).astype(float)
            swir  = src.read(swir_idx).astype(float)

            band_info = (
                f"✅ Band Detection Successful:\n"
                f"BLUE: Band {blue_idx} ({src.descriptions[blue_idx-1] if src.descriptions[blue_idx-1] else 'B2'})\n"
                f"GREEN: Band {green_idx} ({src.descriptions[green_idx-1] if src.descriptions[green_idx-1] else 'B3'})\n"
                f"RED: Band {red_idx} ({src.descriptions[red_idx-1] if src.descriptions[red_idx-1] else 'B4'})\n"
                f"NIR: Band {nir_idx} ({src.descriptions[nir_idx-1] if src.descriptions[nir_idx-1] else 'B8'})\n"
                f"SWIR: Band {swir_idx} ({src.descriptions[swir_idx-1] if src.descriptions[swir_idx-1] else 'B11'})"
            )

            image_info_label.config(text=band_info, fg=COLORS['success'])

        # ============ CALCULATE ALL THREE INDICES ============
        set_status("🧮 Calculating NDSI...", "warning")
        root.update()

        # NDSI = (Green - SWIR) / (Green + SWIR)
        with np.errstate(divide='ignore', invalid='ignore'):  # 1️⃣3️⃣
            ndsi = (green - swir) / (green + swir)
            ndsi = np.where(np.isfinite(ndsi), ndsi, np.nan)

        set_status("🧮 Calculating NDWI...", "warning")
        root.update()

        # NDWI = (Green - NIR) / (Green + NIR)
        with np.errstate(divide='ignore', invalid='ignore'):  # 1️⃣3️⃣
            ndwi = (green - nir) / (green + nir)
            ndwi = np.where(np.isfinite(ndwi), ndwi, np.nan)

        set_status("🧮 Calculating NDVI...", "warning")
        root.update()

        # NDVI = (NIR - Red) / (NIR + Red)
        with np.errstate(divide='ignore', invalid='ignore'):  # 1️⃣3️⃣
            ndvi = (nir - red) / (nir + red)
            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

        # ============ CALCULATE STATISTICS ============
        stats = {}
        for name, data in [('NDSI', ndsi), ('NDWI', ndwi), ('NDVI', ndvi)]:
            valid_data = data[~np.isnan(data)]
            if valid_data.size > 0:
                stats[name] = {
                    'mean': np.mean(valid_data),
                    'min':  np.min(valid_data),
                    'max':  np.max(valid_data),
                    'std':  np.std(valid_data)
                }
            else:
                stats[name] = {'mean': 0, 'min': 0, 'max': 0, 'std': 0}

        # ============ OUTPUT FOLDER ============
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if not output_folder_path:
            output_folder_path = os.path.join(os.path.dirname(image_path), f"{base_name}_ALL_INDICES")
            os.makedirs(output_folder_path, exist_ok=True)

        # ============ SAVE INDIVIDUAL TIFF FILES ============
        set_status("💾 Saving individual NDSI TIFF...", "warning")
        root.update()

        # Update profile for single-band outputs
        single_profile = profile.copy()
        single_profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan
        )

        # Save NDSI
        ndsi_path = os.path.join(output_folder_path, f"{base_name}_NDSI_processed.tiff")
        with rasterio.open(ndsi_path, 'w', **single_profile) as dst:
            dst.write(ndsi.astype(np.float32), 1)
            dst.set_band_description(1, 'NDSI (Snow Index)')

        # Save NDWI
        set_status("💾 Saving individual NDWI TIFF...", "warning")
        root.update()
        ndwi_path = os.path.join(output_folder_path, f"{base_name}_NDWI_processed.tiff")
        with rasterio.open(ndwi_path, 'w', **single_profile) as dst:
            dst.write(ndwi.astype(np.float32), 1)
            dst.set_band_description(1, 'NDWI (Water Index)')

        # Save NDVI
        set_status("💾 Saving individual NDVI TIFF...", "warning")
        root.update()
        ndvi_path = os.path.join(output_folder_path, f"{base_name}_NDVI_processed.tiff")
        with rasterio.open(ndvi_path, 'w', **single_profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)
            dst.set_band_description(1, 'NDVI (Vegetation Index)')

        # ============ CREATE 8-BAND COMPOSITE TIFF ============
        set_status("🎨 Creating 8-band composite TIFF...", "warning")
        root.update()

        composite_tiff_path = os.path.join(output_folder_path, f"{base_name}_processed_composite.tiff")

        # Update profile for 8 bands
        composite_profile = profile.copy()
        composite_profile.update(
            dtype=rasterio.float32,
            count=8,
            nodata=np.nan
        )

        # Write 8-band composite TIFF with proper band order
        with rasterio.open(composite_tiff_path, 'w', **composite_profile) as dst:
            dst.write((blue / SCALING_FACTOR).astype(np.float32), 1)
            dst.write((green / SCALING_FACTOR).astype(np.float32), 2)
            dst.write((red / SCALING_FACTOR).astype(np.float32), 3)
            dst.write((nir / SCALING_FACTOR).astype(np.float32), 4)
            dst.write((swir / SCALING_FACTOR).astype(np.float32), 5)
            dst.write(ndsi.astype(np.float32), 6)
            dst.write(ndwi.astype(np.float32), 7)
            dst.write(ndvi.astype(np.float32), 8)

            dst.set_band_description(1, 'BLUE (B2)')
            dst.set_band_description(2, 'GREEN (B3)')
            dst.set_band_description(3, 'RED (B4)')
            dst.set_band_description(4, 'NIR (B8)')
            dst.set_band_description(5, 'SWIR (B11)')
            dst.set_band_description(6, 'NDSI (Snow Index)')
            dst.set_band_description(7, 'NDWI (Water Index)')
            dst.set_band_description(8, 'NDVI (Vegetation Index)')

        set_status("✓ All files saved: 3 individual TIFFs + 8-band composite TIFF", "success")

        # Update image info
        image_info_label.config(
            text=f"✅ NDSI: {stats['NDSI']['mean']:.3f} | NDWI: {stats['NDWI']['mean']:.3f} | NDVI: {stats['NDVI']['mean']:.3f}",
            fg=COLORS['success']
        )

        messagebox.showinfo(
            "Success",
            f"All 3 indices calculated and saved!\n\n"
            f"Time saved by batch processing!\n\n"
            f"Location: {output_folder_path}\n\n"
            f"Individual TIFF Files (3):\n"
            f"• {base_name}_NDSI_processed.tiff\n"
            f"• {base_name}_NDWI_processed.tiff\n"
            f"• {base_name}_NDVI_processed.tiff\n\n"
            f"Composite Files (1):\n"
            f"• {base_name}_processed_composite.tiff (8-band)\n\n"
            f"Total: 4 files created"
        )

    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate all indices:\n{str(e)}")
        set_status("❌ Calculation failed", "error")
        import traceback
        traceback.print_exc()


# ---------------- RENDER RESULT ON CANVAS ----------------
def render_result_on_canvas(data, calc_type):
    """Render the calculated index result onto preview_canvas with a colormap."""
    try:
        import matplotlib.pyplot as plt
        import io

        valid = ~np.isnan(data)

        # Choose colormap per index type
        if calc_type == 'NDSI':
            cmap = plt.cm.Blues
            vmin, vmax = 0.0, 1.0
            display_data = np.clip(data, vmin, vmax)
        elif calc_type == 'NDWI':
            cmap = plt.cm.Blues
            vmin, vmax = 0.0, 1.0
            display_data = np.clip(data, vmin, vmax)
        else:  # NDVI
            cmap = plt.cm.Greens
            vmin, vmax = 0.0, 1.0
            display_data = np.clip(data, vmin, vmax)

        # Normalise 0-1
        if vmax > vmin:
            normalized = (display_data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(display_data)

        # NaN → 0 (black in final image)
        normalized[~valid] = 0.0

        # Apply colormap → RGBA 0-1
        rgba = cmap(normalized)

        # Force NaN pixels to pure black
        rgba[~valid] = [0.0, 0.0, 0.0, 1.0]

        # Convert to uint8 RGB
        rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)

        pil_img = PILImage.fromarray(rgb_uint8, mode='RGB')

        # Scale to fit canvas
        canvas_width = preview_canvas.winfo_width() or 600
        canvas_height = preview_canvas.winfo_height() or 500
        img_width, img_height = pil_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
        pil_img = pil_img.resize(
            (int(img_width * scale), int(img_height * scale)),
            PILImage.LANCZOS
        )

        photo = ImageTk.PhotoImage(pil_img)
        preview_canvas.delete("all")
        preview_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=photo, anchor='center'
        )
        preview_canvas.image = photo   # prevent GC
        preview_label.place_forget()

    except Exception as e:
        print(f"Render result error: {e}")


# ---------------- RUN CALCULATION ----------------
def run_calculation():
    global result_data, profile, green, swir, output_folder_path

    # Check if folder is loaded
    if not folder_path:
        messagebox.showerror("Error", "Please select a folder first!")
        return

    # Check if an image is selected from the listbox
    selection = image_listbox.curselection()

    calc_type = calculation_var.get()

    # Check if "All Indices" is selected
    if calc_type == "All Indices":
        if selection:
            # Image selected → Run batch for SINGLE image only
            calculate_all_indices()
        else:
            # No image selected → Run batch for ALL images in folder
            batch_process_all_images()
        return

    # SINGLE-INDEX CALCULATION (NDSI, NDWI, or NDVI)
    
    # Check if image is selected
    if not selection or not image_path:
        # No image selected → Batch process all images with single index
        batch_process_single_index(calc_type)
        return

    # Image selected → Process single image with single index
    try:
        set_status(f"🔍 Detecting bands for {calc_type}...", "warning")
        root.update()

        with rasterio.open(image_path) as src:
            profile = src.profile

            # Smart band detection (uses cache internally via get_required_bands → _cached_detect_band)
            band_result = get_required_bands(src, calc_type)

            if not band_result['success']:
                messagebox.showerror(
                    "Band Detection Failed",
                    f"Could not detect required bands for {calc_type}.\n\n"
                    f"{band_result['info']}\n\n"
                    f"Image has {src.count} bands with descriptions:\n"
                    f"{src.descriptions}\n\n"
                    f"Please ensure your GEE export has proper band names."
                )
                set_status("❌ Band detection failed", "error")
                return

            # Extract band data
            bands = band_result['bands']

            # Calculate index based on type
            if calc_type == 'NDSI':
                band1_data = bands['GREEN']['data']
                band2_data = bands['SWIR']['data']
            elif calc_type == 'NDWI':
                band1_data = bands['GREEN']['data']
                band2_data = bands['NIR']['data']
            elif calc_type == 'NDVI':
                band1_data = bands['NIR']['data']
                band2_data = bands['RED']['data']

            # Show which bands are being used
            set_status(f"⏳ Calculating {calc_type}... ({band_result['info']})", "warning")
            root.update()

        # Calculate index
        denom = band1_data + band2_data

        with np.errstate(divide='ignore', invalid='ignore'):  # 1️⃣3️⃣
            result_data = (band1_data - band2_data) / denom

        result_data[denom == 0] = np.nan
        result_data = np.clip(result_data, -1, 1)

        # Statistics calculation
        total_pixels = result_data.size
        valid_mask = ~np.isnan(result_data)
        valid_pixels = np.sum(valid_mask)
        nodata_pixels = total_pixels - valid_pixels

        # All indices now use >= for positive classification
        threshold_mask = result_data >= ndsi_threshold
        positive_pixels = np.sum(threshold_mask & valid_mask)
        negative_pixels = valid_pixels - positive_pixels

        positive_percentage = (positive_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        negative_percentage = (negative_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        nodata_percentage  = (nodata_pixels  / total_pixels) * 100 if total_pixels > 0 else 0

        # Value distribution
        range1 = np.sum((result_data >= -1)   & (result_data < -0.5) & valid_mask)
        range2 = np.sum((result_data >= -0.5) & (result_data < 0)    & valid_mask)
        range3 = np.sum((result_data >= 0)    & (result_data < 0.5)  & valid_mask)
        range4 = np.sum((result_data >= 0.5)  & (result_data <= 1)   & valid_mask)

        # Class names
        if calc_type == "NDSI":
            class1_name, class2_name = "Snow", "Non-snow"
        elif calc_type == "NDWI":
            class1_name, class2_name = "Water", "Non-water"
        elif calc_type == "NDVI":
            class1_name, class2_name = "Vegetation", "Non-vegetation"

        calc_data = {
            'filename':            os.path.basename(image_path),
            'index_type':          calc_type,
            'threshold':           ndsi_threshold,
            'min':                 np.nanmin(result_data),
            'max':                 np.nanmax(result_data),
            'mean':                np.nanmean(result_data),
            'std':                 np.nanstd(result_data),
            'total_pixels':        total_pixels,
            'valid_pixels':        valid_pixels,
            'nodata_pixels':       nodata_pixels,
            'positive_pixels':     positive_pixels,
            'negative_pixels':     negative_pixels,
            'positive_percentage': positive_percentage,
            'negative_percentage': negative_percentage,
            'nodata_percentage':   nodata_percentage,
            'range1':              range1,
            'range2':              range2,
            'range3':              range3,
            'range4':              range4,
            'band_info':           band_result['info'],
            'class1_name':         class1_name,
            'class2_name':         class2_name
        }

        set_status(f"✓ {calc_type} calculated successfully ({band_result['info']})", "success")

        # Update image info with calculation results
        image_info_label.config(
            text=f"✅ {calc_type}: Min={calc_data['min']:.3f}, Max={calc_data['max']:.3f}, Mean={calc_data['mean']:.3f}",
            fg=COLORS['success']
        )

        # Render the calculated index on the preview canvas
        render_result_on_canvas(result_data, calc_type)

        # Save results if output folder is selected
        print(f"DEBUG run_calc: calc_type={calc_type}, output_folder_path={output_folder_path}, result_data shape={result_data.shape if result_data is not None else None}")
        if output_folder_path:
            save_results(calc_data)
        else:
            messagebox.showinfo(
                "Calculation Complete",
                f"{calc_type} calculated successfully!\n\n"
                f"Threshold: {ndsi_threshold:.2f}\n"
                f"{class1_name}: {positive_percentage:.2f}%\n"
                f"{class2_name}: {negative_percentage:.2f}%\n\n"
                f"Select an output folder to save results."
            )

    except Exception as e:
        messagebox.showerror("Error", f"Calculation failed:\n{str(e)}")
        set_status("❌ Calculation error", "error")
        import traceback
        traceback.print_exc()


# ---------------- SAVE RESULTS ----------------
def save_results(calc_data):
    """Save TIFF and PDF report"""
    global result_data, profile

    # 1️⃣2️⃣ guard clauses
    if result_data is None:
        messagebox.showwarning("No Data", "No calculation results to save.")
        return
    if not output_folder_path:
        messagebox.showinfo("No Output Folder", "Please select an output folder first.")
        return

    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        calc_type = calc_data['index_type']

        # Save TIFF with _processed naming convention
        out_tif = os.path.join(output_folder_path, f"{base_name}_{calc_type}_processed.tiff")

        temp_profile = profile.copy()
        temp_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        print(f"DEBUG save_results: Writing TIFF to: {out_tif}")
        with rasterio.open(out_tif, 'w', **temp_profile) as dst:
            dst.write(result_data.astype(np.float32), 1)
        print(f"DEBUG save_results: TIFF written. File exists: {os.path.exists(out_tif)}, size: {os.path.getsize(out_tif)} bytes")


        # Create PDF report with _processed_report naming convention
        pdf_path = os.path.join(output_folder_path, f"{base_name}_{calc_type}_processed_report.pdf")

        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph(f"<b>{calc_type} Calculation Report</b><br/><br/>", styles['Title']))

        # Add file information
        story.append(Paragraph(f"<b>File:</b> {calc_data['filename']}<br/>", styles['Normal']))
        story.append(Paragraph(f"<b>Index Type:</b> {calc_type}<br/>", styles['Normal']))
        story.append(Paragraph(f"<b>Band Info:</b> {calc_data['band_info']}<br/>", styles['Normal']))
        story.append(Paragraph(f"<b>Threshold:</b> {calc_data['threshold']:.2f}<br/><br/>", styles['Normal']))

        # Add statistics
        story.append(Paragraph("<b>Statistics:</b><br/>", styles['Heading2']))
        story.append(Paragraph(f"Min: {calc_data['min']:.4f}<br/>", styles['Normal']))
        story.append(Paragraph(f"Max: {calc_data['max']:.4f}<br/>", styles['Normal']))
        story.append(Paragraph(f"Mean: {calc_data['mean']:.4f}<br/>", styles['Normal']))
        story.append(Paragraph(f"Std Dev: {calc_data['std']:.4f}<br/><br/>", styles['Normal']))

        # Add pixel counts
        story.append(Paragraph("<b>Pixel Classification:</b><br/>", styles['Heading2']))
        story.append(Paragraph(f"Total Pixels: {calc_data['total_pixels']:,}<br/>", styles['Normal']))
        story.append(Paragraph(
            f"{calc_data['class1_name']}: {calc_data['positive_pixels']:,} ({calc_data['positive_percentage']:.2f}%)<br/>",
            styles['Normal']
        ))
        story.append(Paragraph(
            f"{calc_data['class2_name']}: {calc_data['negative_pixels']:,} ({calc_data['negative_percentage']:.2f}%)<br/>",
            styles['Normal']
        ))
        story.append(Paragraph(f"No Data: {calc_data['nodata_pixels']:,} ({calc_data['nodata_percentage']:.2f}%)<br/><br/>", styles['Normal']))


        doc.build(story)

        set_status(f"✓ Saved: {base_name}_{calc_type}_processed.tiff + report", "success")

        # Show success message with file names
        messagebox.showinfo(
            "Success",
            f"Files saved successfully!\n\n"
            f"Location: {output_folder_path}\n\n"
            f"Files:\n"
            f"• {base_name}_{calc_type}_processed.tiff\n"
            f"• {base_name}_{calc_type}_processed_report.pdf"
        )

    except Exception as e:
        import traceback
        print(f"DEBUG save_results EXCEPTION: {e}")
        traceback.print_exc()
        messagebox.showerror("Save Error", f"Failed to save files:\n{str(e)}")
        set_status("❌ Failed to save outputs", "error")


# ============================================================
# HELPER: draw a card (1️⃣ replaces LabelFrame)
# Returns (outer_frame, inner_frame)
# ============================================================
def _make_card(parent, title_text, title_icon=""):
    """
    Create a modern card widget to replace LabelFrame.
    Returns the inner frame that children should be packed into.
    """
    # Outer frame – border
    outer = tk.Frame(parent, bg=COLORS['card_border'], bd=0, highlightthickness=0)

    # Inner frame – card body
    inner = tk.Frame(outer, bg=COLORS['card_bg'], bd=0, highlightthickness=0)
    inner.pack(fill='both', expand=True, padx=1, pady=1)   # 1 px border effect

    # Header label
    header = tk.Label(
        inner,
        text=f"{title_icon} {title_text}" if title_icon else title_text,
        font=_FONT_SECTION,
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w',
        padx=18,
        pady=10
    )
    header.pack(fill='x')

    # Thin separator line under header
    sep = tk.Frame(inner, bg=COLORS['card_border'], height=1)
    sep.pack(fill='x', padx=18)

    # Content frame (padded)
    content = tk.Frame(inner, bg=COLORS['card_bg'])
    content.pack(fill='both', expand=True, padx=18, pady=(12, 14))

    return outer, content


# ============================================================
# GUI SETUP
# ============================================================
root = tk.Tk()
root.title("Raster Index Calculator – Professional Edition")

# Get screen dimensions
screen_width  = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Platform-specific window setup
if IS_WINDOWS:
    root.state('zoomed')
elif IS_MAC:
    try:
        root.attributes('-zoomed', True)
    except Exception:
        root.geometry(f"{screen_width}x{screen_height}+0+0")
else:  # Linux
    try:
        root.attributes('-zoomed', True)
    except Exception:
        root.geometry(f"{screen_width}x{screen_height}+0+0")

root.config(bg=COLORS['background'])

# ── macOS fix: force all tk widgets to honour explicit bg/fg ──
# On macOS, tk.Button ignores bg/fg kwargs (especially inside ttk.Notebook).
# tk_setPalette sets the X-resource defaults that Tk actually reads on Mac.
if IS_MAC:
    root.tk_setPalette(
        background=COLORS['background'],
        foreground=COLORS['text'],
        activeBackground=COLORS['secondary_hover'],
        activeForeground='#ffffff',
        highlightColor=COLORS['card_border'],
        highlightBackground=COLORS['card_border'],
        selectColor=COLORS['listbox_sel'],
        selectBackground=COLORS['listbox_bg'],
        selectForeground=COLORS['text'],
        insertBackground=COLORS['text'],
        troughColor=COLORS['card_bg']
    )

# ──── Main container ────
main_container = tk.Frame(root, bg=COLORS['background'])
main_container.pack(fill='both', expand=True, padx=28, pady=18)

# ──── Header (5️⃣ typography) ────
header_frame = tk.Frame(main_container, bg=COLORS['card_bg'], bd=0, highlightthickness=0)
header_frame.pack(fill='x', pady=(0, 22))
# subtle bottom border on header
header_border = tk.Frame(header_frame, bg=COLORS['card_border'], height=1)

title_label = tk.Label(
    header_frame,
    text="🛰️  Raster Index Calculator",
    font=_FONT_TITLE,
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    pady=14
)
title_label.pack()

subtitle_label = tk.Label(
    header_frame,
    text="Professional GeoTIFF Analysis Tool with Smart Band Detection",
    font=_FONT_SUB,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted']
)
subtitle_label.pack(pady=(0, 6))
header_border.pack(fill='x')

# ──── Two-column content area ────
content_frame = tk.Frame(main_container, bg=COLORS['background'])
content_frame.pack(fill='both', expand=True)

content_frame.grid_rowconfigure(0, weight=1)
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)

# ──── Left column (non-scrollable) ────
left_column = tk.Frame(content_frame, bg=COLORS['background'], bd=0, highlightthickness=0)
left_column.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

# ──── Right column ────
right_column = tk.Frame(content_frame, bg=COLORS['background'], bd=0, highlightthickness=0)
right_column.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

# ============================================================
# LEFT COLUMN CARDS
# ============================================================

# ─── 1. FOLDERS CARD ───
folders_card_outer, folders_card = _make_card(left_column, "Folders", "📁")
folders_card_outer.pack(fill='x', pady=(0, 12))

upload_info = tk.Label(
    folders_card,
    text="Select input and output folders",
    font=_FONT_SMALL,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w'
)
upload_info.pack(fill='x', pady=(0, 8))

# Input folder row
input_folder_label = tk.Label(
    folders_card,
    text="Input Folder:",
    font=("Segoe UI", 9, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    anchor='w'
)
input_folder_label.pack(fill='x')

folder_label = tk.Label(
    folders_card,
    text="No folder selected",
    font=_FONT_BODY,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w'
)
folder_label.pack(fill='x', pady=(2, 0))

folder_info_label = tk.Label(
    folders_card,
    text="",
    font=_FONT_SMALL,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w'
)
folder_info_label.pack(fill='x', pady=(0, 8))

# Folder buttons row (3️⃣ secondary style)
upload_buttons_frame = tk.Frame(folders_card, bg=COLORS['card_bg'])
upload_buttons_frame.pack(fill='x', pady=(0, 4))

btn_input_folder = tk.Button(
    upload_buttons_frame,
    text="📂  Select Input Folder",
    font=_FONT_BTN,
    bg=COLORS['secondary_bg'],
    fg='#ffffff',
    activebackground=COLORS['secondary_hover'],
    activeforeground='#ffffff',
    pady=10,
    cursor="hand2",
    relief='flat',
    bd=0,
    highlightthickness=2,
    highlightbackground=COLORS['secondary_border'],
    highlightcolor=COLORS['secondary_border'],
    command=upload_folder
)
btn_input_folder.pack(side='left', fill='x', expand=True, padx=(0, 5))

btn_output_folder = tk.Button(
    upload_buttons_frame,
    text="💾  Select Output Folder",
    font=_FONT_BTN,
    bg=COLORS['secondary_bg'],
    fg='#ffffff',
    activebackground=COLORS['secondary_hover'],
    activeforeground='#ffffff',
    pady=10,
    cursor="hand2",
    relief='flat',
    bd=0,
    highlightthickness=2,
    highlightbackground=COLORS['secondary_border'],
    highlightcolor=COLORS['secondary_border'],
    command=select_output_folder_ndsi  # 🔧 FIXED: Now calls NDSI-specific function
)
btn_output_folder.pack(side='right', fill='x', expand=True, padx=(5, 0))

# Output folder display
output_folder_label = tk.Label(
    folders_card,
    text="Output Folder:",
    font=("Segoe UI", 9, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    anchor='w'
)
output_folder_label.pack(fill='x', pady=(10, 0))

output_folder_display = tk.Label(
    folders_card,
    text="Not selected",
    font=_FONT_BODY,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w'
)
output_folder_display.pack(fill='x', pady=(2, 0))

# Listbox (initially hidden)
listbox_frame = tk.Frame(folders_card, bg=COLORS['card_bg'])

listbox_label = tk.Label(
    listbox_frame,
    text="Images in folder:",
    font=("Segoe UI", 9, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    anchor='w'
)
listbox_label.pack(fill='x', pady=(0, 5))

listbox_scroll_frame = tk.Frame(listbox_frame, bg=COLORS['card_bg'])
listbox_scroll_frame.pack(fill='both', expand=True)

image_listbox_scrollbar = tk.Scrollbar(listbox_scroll_frame)
image_listbox_scrollbar.pack(side='right', fill='y')

image_listbox = tk.Listbox(
    listbox_scroll_frame,
    height=4,
    font=_FONT_SMALL,
    bg=COLORS['listbox_bg'],
    fg=COLORS['text'],
    selectbackground=COLORS['listbox_sel'],
    selectforeground='#ffffff',
    yscrollcommand=image_listbox_scrollbar.set,
    relief='flat',
    bd=0,
    highlightthickness=1,
    highlightbackground=COLORS['card_border'],
    activestyle='none'
)
image_listbox.pack(side='left', fill='both', expand=True)
image_listbox_scrollbar.config(command=image_listbox.yview)
image_listbox.bind('<<ListboxSelect>>', load_selected_image)

# Add button to clear selection (for batch processing all images)
def clear_image_selection():
    """Clear listbox selection to enable batch processing mode"""
    image_listbox.selection_clear(0, tk.END)
    global image_path
    image_path = None
    file_name_label.config(
        text="ALL IMAGES SELECTED(Batch Mode)",
        fg=COLORS['warning']
    )
    file_info_label.config(text="Run 'All Indices' to process all images in folder", fg=COLORS['text_muted'])
    set_status("🔄 Batch mode: Run 'All Indices' to process entire folder", "info")

clear_selection_btn = tk.Button(
    listbox_frame,
    text="🔄 Clear Selection (Batch Mode)",
    font=_FONT_SMALL,
    bg=COLORS['secondary_bg'],
    fg='#ffffff',
    activebackground=COLORS['secondary_hover'],
    activeforeground='#ffffff',
    pady=6,
    cursor="hand2",
    relief='flat',
    bd=0,
    command=clear_image_selection
)
clear_selection_btn.pack(fill='x', pady=(6, 0))

# ─── 2. CURRENT IMAGE CARD ───
image_card_outer, image_card = _make_card(left_column, "Current Image", "📄")
image_card_outer.pack(fill='x', pady=(0, 12))

file_name_label = tk.Label(
    image_card,
    text="No image selected",
    font=_FONT_BODY,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w',
    wraplength=400
)
file_name_label.pack(fill='x', pady=(0, 4))

file_info_label = tk.Label(
    image_card,
    text="",
    font=_FONT_SMALL,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w'
)
file_info_label.pack(fill='x')

# ─── 3 & 4. CALCULATION AND ACTIONS (SIDE-BY-SIDE) ───
calc_action_frame = tk.Frame(left_column, bg=COLORS['background'])
calc_action_frame.pack(fill='x', pady=(0, 12))

# Configure grid for equal width (50-50 split)
calc_action_frame.grid_columnconfigure(0, weight=1, uniform='calc_cols')
calc_action_frame.grid_columnconfigure(1, weight=1, uniform='calc_cols')

# Left: Calculation
calc_left = tk.Frame(calc_action_frame, bg=COLORS['background'])
calc_left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))

# Right: Actions
calc_right = tk.Frame(calc_action_frame, bg=COLORS['background'])
calc_right.grid(row=0, column=1, sticky='nsew', padx=(6, 0))

# ─── 3. CALCULATION CARD (in left half) ───
calc_card_outer, calc_card = _make_card(calc_left, "Calculation", "⚙️")
calc_card_outer.pack(fill='both', expand=True)

calc_type_label = tk.Label(
    calc_card,
    text="Index Type:",
    font=("Segoe UI", 10, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    anchor='w'
)
calc_type_label.pack(fill='x', pady=(0, 4))

# Style the combobox via a ttk Style
_style = ttk.Style()
_style.theme_use('classic')
_style.configure("TCombobox",
    background=COLORS['listbox_bg'],
    foreground=COLORS['text'],
    insertcolor=COLORS['text'],
    arrowcolor=COLORS['primary'],
    selectbackground=COLORS['listbox_bg'],
    selectforeground=COLORS['text']
)
_style.map("TCombobox",
    background=[('readonly', COLORS['listbox_bg']),
                ('!disabled', COLORS['listbox_bg'])],
    foreground=[('readonly', COLORS['text']),
                ('!disabled', COLORS['text'])],
    selectbackground=[('readonly', COLORS['listbox_bg'])],
    selectforeground=[('readonly', COLORS['text'])]
)

calculation_var = tk.StringVar(value="NDSI")
calc_dropdown = ttk.Combobox(
    calc_card,
    textvariable=calculation_var,
    values=["NDSI", "NDWI", "NDVI", "All Indices"],
    state='readonly',
    font=_FONT_BODY
)
calc_dropdown.pack(fill='x')

# Dropdown change handler
def on_calculation_change(event):
    calc_type = calculation_var.get()
    if calc_type == "NDSI":
        threshold_label.config(text="NDSI Threshold:")
        threshold_slider.config(from_=0.0, to=1.0, state='normal')
        threshold_slider.set(0.4)
    elif calc_type == "NDWI":
        threshold_label.config(text="NDWI Threshold:")
        threshold_slider.config(from_=-1.0, to=1.0, state='normal')
        threshold_slider.set(0.3)
    elif calc_type == "NDVI":
        threshold_label.config(text="NDVI Threshold:")
        threshold_slider.config(from_=-1.0, to=1.0, state='normal')
        threshold_slider.set(0.2)
    elif calc_type == "All Indices":
        threshold_label.config(text="Thresholds (Auto):")
        threshold_slider.config(state='disabled')

calc_dropdown.bind('<<ComboboxSelected>>', on_calculation_change)

# 8️⃣ Threshold row: label + badge on same line
threshold_frame = tk.Frame(calc_card, bg=COLORS['card_bg'])
threshold_frame.pack(fill='x', pady=(12, 0))

threshold_header = tk.Frame(threshold_frame, bg=COLORS['card_bg'])
threshold_header.pack(fill='x')

threshold_label = tk.Label(
    threshold_header,
    text="NDSI Threshold:",
    font=("Segoe UI", 10, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text']
)
threshold_label.pack(side='left')

# Badge – colored pill showing the current value
threshold_value_label = tk.Label(
    threshold_header,
    text="0.40",
    font=_FONT_BADGE,
    bg=COLORS['primary'],          # accent colour badge
    fg='#ffffff',
    padx=8,
    pady=2,
    bd=0,
    relief='flat'
)
threshold_value_label.pack(side='right')

threshold_slider = tk.Scale(
    threshold_frame,
    from_=0.0,
    to=1.0,
    resolution=0.05,
    orient='horizontal',
    command=on_threshold_change,
    bg=COLORS['card_bg'],
    fg=COLORS['text'],
    highlightthickness=0,
    troughcolor=COLORS['card_border'],
    activebackground=COLORS['primary'],
    sliderrelief='flat',
    showvalue=False                # value shown in badge instead
)
threshold_slider.set(0.4)
threshold_slider.pack(fill='x', pady=(6, 0))

# ─── 4. ACTIONS CARD (in right half) ───
action_card_outer, action_card = _make_card(calc_right, "Actions", "🚀")
action_card_outer.pack(fill='both', expand=True)

# 3️⃣ Primary button – accent colour, clearly the main CTA
btn_run = tk.Button(
    action_card,
    text="▶  Run Calculation",
    font=_FONT_BTN,
    bg=COLORS['primary_disabled'],   # starts disabled (7️⃣)
    fg="#ffffff",
    activebackground=COLORS['primary_hover'],
    activeforeground="#ffffff",
    pady=12,
    state='disabled',
    cursor="hand2",
    relief='flat',
    bd=0,
    highlightthickness=2,
    highlightbackground='#4a5568',
    highlightcolor='#4a5568',
    command=run_calculation
)
btn_run.pack(fill='x')
_btn_run = btn_run   # 7️⃣ assign module-level ref

# Hover effects for run button
def _on_enter_run(e):
    if btn_run['state'] == 'normal':
        btn_run.config(bg=COLORS['primary_hover'], highlightbackground='#9bb8fd', highlightcolor='#9bb8fd')

def _on_leave_run(e):
    if btn_run['state'] == 'normal':
        btn_run.config(bg=COLORS['primary'], highlightbackground='#7aa3fc', highlightcolor='#7aa3fc')

btn_run.bind("<Enter>", _on_enter_run)
btn_run.bind("<Leave>", _on_leave_run)

# ============================================================
# RIGHT COLUMN – TABBED PANEL (9️⃣)
# ============================================================

# Style the Notebook tabs
_style.configure("TNotebook", background=COLORS['background'], borderwidth=0)
_style.configure("TNotebook.Tab",
    background=COLORS['card_bg'],
    foreground=COLORS['text_muted'],
    padding=[18, 8],
    font=("Segoe UI", 10, "bold"),
    borderwidth=0
)
_style.map("TNotebook.Tab",
    background=[("selected", COLORS['card_bg'])],
    foreground=[("selected", COLORS['primary'])]
)

notebook = ttk.Notebook(right_column)
notebook.pack(fill='both', expand=True)

# ─── Tab 1: Image Preview ───
preview_tab = tk.Frame(notebook, bg=COLORS['card_bg'])
notebook.add(preview_tab, text='🖼️  Image Preview')

# Card inside preview tab
preview_card_outer, preview_card = _make_card(preview_tab, "Preview", "📸")
preview_card_outer.pack(fill='both', expand=True, padx=6, pady=6)

# 4️⃣ Dark canvas for raster preview
preview_canvas = tk.Canvas(
    preview_card,
    bg=COLORS['canvas_bg'],
    highlightthickness=1,
    highlightbackground=COLORS['card_border']
)
preview_canvas.pack(fill='both', expand=True)

# Placeholder label on canvas
preview_label = tk.Label(
    preview_canvas,
    text="No image loaded\n\n📂 Select an image to see preview",
    font=("Segoe UI", 12),
    bg=COLORS['canvas_bg'],
    fg=COLORS['text_muted'],
    justify='center'
)
preview_label.place(relx=0.5, rely=0.5, anchor='center')

# Info line below canvas
image_info_label = tk.Label(
    preview_card,
    text="",
    font=_FONT_SMALL,
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    justify='left',
    anchor='w'
)
image_info_label.pack(fill='x', pady=(6, 0))

# ─── Tab 2: Load Model (lazy-built on first select) ───
model_tab = tk.Frame(notebook, bg=COLORS['card_bg'])
notebook.add(model_tab, text='🤖  Load Model')

# These will be assigned inside _build_model_tab() on first switch
model_filename_display = None
_model_tab_built       = False

def save_prediction_result():
    """Save the prediction result text to a file."""
    global model_result_label, model_input_image_path
    
    if model_result_label is None:
        messagebox.showwarning("No Result", "No prediction result to save!")
        return
    
    result_text = model_result_label.cget("text")
    if result_text in ["No prediction yet", ""]:
        messagebox.showwarning("No Result", "Please run a prediction first!")
        return
    
    try:
        # Ask user where to save
        save_path = filedialog.asksaveasfilename(
            title="Save Prediction Result",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile=f"prediction_result_{os.path.splitext(os.path.basename(model_input_image_path))[0]}.txt" if model_input_image_path else "prediction_result.txt"
        )
        
        if save_path:
            # Save the text
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            messagebox.showinfo("Saved", f"Prediction result saved to:\n{save_path}")
            set_status(f"✅ Result saved: {os.path.basename(save_path)}", "success")
    
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save result:\n{str(e)}")
        set_status("❌ Failed to save result", "error")

def _build_model_tab():
    """Build model-tab widgets exactly once, on first tab-switch to it."""
    global model_filename_display, _model_tab_built
    global model_result_label
    global window_size, stride, window_size_var, stride_var
    global model_selected_image_listbox
    global model_folder_label, model_folder_info_label, model_output_folder_label, model_output_folder_info
    if _model_tab_built:
        return
    _model_tab_built = True

    # Main container frame for model tab (NON-SCROLLABLE)
    model_main_frame = tk.Frame(model_tab, bg=COLORS['card_bg'])
    model_main_frame.pack(fill='both', expand=True, padx=6, pady=6)

    # ===== SECTION 1: Model Upload =====
    model_card_outer, model_card = _make_card(model_main_frame, "Machine Learning Model", "🤖")
    model_card_outer.pack(fill='x', pady=(0, 12))

    tk.Label(
        model_card,
        text="Upload your trained Keras model (.keras, .h5) for classification",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        justify='left',
        wraplength=580,
        anchor='w'
    ).pack(fill='x', pady=(0, 14))

    tk.Label(
        model_card,
        text="Selected Model:",
        font=("Segoe UI", 10, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(fill='x', pady=(0, 4))

    model_filename_display = tk.Label(
        model_card,
        text="No model loaded",
        font=_FONT_BODY,
        bg=COLORS['listbox_bg'],
        fg=COLORS['text_muted'],
        anchor='w',
        padx=14,
        pady=10,
        relief='flat',
        bd=0,
        highlightthickness=1,
        highlightbackground=COLORS['card_border']
    )
    model_filename_display.pack(fill='x', pady=(0, 12))

    # Buttons frame for side-by-side layout
    buttons_frame = tk.Frame(model_card, bg=COLORS['card_bg'])
    buttons_frame.pack(fill='x', pady=(0, 10))
    
    tk.Button(
        buttons_frame,
        text="📤  Upload Model (.keras/.h5)",
        font=_FONT_SMALL,
        bg=COLORS['primary'],
        fg='#ffffff',
        activebackground=COLORS['primary_hover'],
        activeforeground='#ffffff',
        pady=8,
        cursor="hand2",
        relief='flat',
        bd=0,
        highlightthickness=1,
        highlightbackground='#7aa3fc',
        highlightcolor='#7aa3fc',
        command=upload_model
    ).pack(side='left', fill='x', expand=True, padx=(0, 5))
    
    tk.Button(
        buttons_frame,
        text="📊  Model Details",
        font=_FONT_SMALL,
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=8,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=show_model_details
    ).pack(side='left', fill='x', expand=True, padx=(5, 0))

    # ===== TWO-COLUMN LAYOUT: Sliding Window Settings | Model Prediction =====
    two_column_frame = tk.Frame(model_main_frame, bg=COLORS['background'])
    two_column_frame.pack(fill='both', expand=True, pady=(0, 12))
    
    # Configure grid columns for equal width (50-50 split)
    two_column_frame.grid_columnconfigure(0, weight=1, uniform='model_cols')
    two_column_frame.grid_columnconfigure(1, weight=1, uniform='model_cols')
    
    # LEFT COLUMN - Sliding Window Settings
    left_column = tk.Frame(two_column_frame, bg=COLORS['background'])
    left_column.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
    
    # RIGHT COLUMN - Model Prediction
    right_column = tk.Frame(two_column_frame, bg=COLORS['background'])
    right_column.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
    
    # ===== LEFT: SLIDING WINDOW SETTINGS =====
    window_config_card_outer, window_config_card = _make_card(left_column, "Sliding Window Settings", "🪟")
    window_config_card_outer.pack(fill='both', expand=True)

    tk.Label(
        window_config_card,
        text="Window Size & Stride",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        justify='left',
        wraplength=280,
        anchor='w'
    ).pack(fill='x', pady=(0, 14))

    # Window Size Control
    window_size_frame = tk.Frame(window_config_card, bg=COLORS['card_bg'])
    window_size_frame.pack(fill='x', pady=(0, 10))

    tk.Label(
        window_size_frame,
        text="Window Size:",
        font=("Segoe UI", 10, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(side='left', padx=(0, 10))

    # Already declared global at function start
    window_size_var = tk.IntVar(value=window_size)  # Use current window_size
    window_size_entry = tk.Entry(
        window_size_frame,
        textvariable=window_size_var,
        font=_FONT_BODY,
        bg=COLORS['listbox_bg'],
        fg=COLORS['text'],
        insertbackground=COLORS['text'],
        relief='flat',
        bd=0,
        width=10,
        highlightthickness=1,
        highlightbackground=COLORS['card_border']
    )
    window_size_entry.pack(side='left', padx=(0, 10))

    tk.Label(
        window_size_frame,
        text="px (square window)",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted']
    ).pack(side='left')

    # Stride Control
    stride_frame = tk.Frame(window_config_card, bg=COLORS['card_bg'])
    stride_frame.pack(fill='x', pady=(0, 10))

    tk.Label(
        stride_frame,
        text="Stride:",
        font=("Segoe UI", 10, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(side='left', padx=(0, 54))

    # Already declared global at function start
    stride_var = tk.IntVar(value=stride)  # Use current stride
    stride_entry = tk.Entry(
        stride_frame,
        textvariable=stride_var,
        font=_FONT_BODY,
        bg=COLORS['listbox_bg'],
        fg=COLORS['text'],
        insertbackground=COLORS['text'],
        relief='flat',
        bd=0,
        width=10,
        highlightthickness=1,
        highlightbackground=COLORS['card_border']
    )
    stride_entry.pack(side='left', padx=(0, 10))

    tk.Label(
        stride_frame,
        text="px (overlap = window_size - stride)",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted']
    ).pack(side='left')

    # Apply Button
    def apply_window_settings():
        global window_size, stride, loaded_model
        try:
            new_window_size = window_size_var.get()
            new_stride = stride_var.get()
            
            if new_window_size < 16:
                messagebox.showwarning("Invalid Window Size", "Window size must be at least 16 pixels")
                return
            
            if new_stride < 1:
                messagebox.showwarning("Invalid Stride", "Stride must be at least 1 pixel")
                return
            
            if new_stride > new_window_size:
                messagebox.showwarning("Invalid Stride", "Stride cannot be larger than window size")
                return
            
            # ✅ CRITICAL VALIDATION: Check if window_size matches loaded model's input size
            if loaded_model is not None:
                model_input_size = loaded_model.input_shape[1]
                
                if new_window_size != model_input_size:
                    messagebox.showerror(
                        "❌ Window Size Mismatch!",
                        f"ERROR: Your model expects {model_input_size}x{model_input_size} patches!\n\n"
                        f"You tried to set: {new_window_size}x{new_window_size}\n\n"
                        f"🚫 This will cause INCORRECT PREDICTIONS!\n\n"
                        f"The model was trained on {model_input_size}x{model_input_size} patches.\n"
                        f"Window size MUST match the model's input size.\n\n"
                        f"✅ Correct setting: {model_input_size}x{model_input_size}\n"
                        f"❌ Your setting: {new_window_size}x{new_window_size}\n\n"
                        f"Please use window_size = {model_input_size}"
                    )
                    return
            
            window_size = new_window_size
            stride = new_stride
            
            overlap = window_size - stride
            messagebox.showinfo(
                "✓ Settings Applied",
                f"Window Size: {window_size}x{window_size}\n"
                f"Stride: {stride}\n"
                f"Overlap: {overlap} pixels\n\n"
                f"{'✅ Matches model input size!' if loaded_model and window_size == loaded_model.input_shape[1] else '⚠️ No model loaded yet'}"
            )
            set_status(f"✓ Window settings updated: {window_size}x{window_size}, stride={stride}", "success")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for window size and stride")

    tk.Button(
        window_config_card,
        text="✓  Apply Settings",
        font=_FONT_BTN,
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=10,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=apply_window_settings
    ).pack(fill='x', pady=(0, 15))
    
    # ===== Model Prediction Section =====
    tk.Label(
        window_config_card,
        text="Model Prediction",
        font=("Segoe UI", 11, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(fill='x', pady=(5, 8))
    
    # Wrapper function to show listbox when folder is loaded
    def upload_model_folder_and_show_listbox():
        upload_model_folder()
        if model_image_files:
            listbox_frame.pack(fill='x', pady=(0, 10))  # Fixed: no expand=True
    
    folder_select_frame = tk.Frame(window_config_card, bg=COLORS['card_bg'])
    folder_select_frame.pack(fill='x', pady=(0, 10))
    
    tk.Button(
        folder_select_frame,
        text="📁  Select Input Folder",
        font=_FONT_BTN,
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=10,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=upload_model_folder_and_show_listbox
    ).pack(fill='x', pady=(0, 5))
    
    tk.Button(
        folder_select_frame,
        text="💾  Select Output Folder",
        font=_FONT_BTN,
        bg=COLORS['secondary_bg'],
        fg='#ffffff',
        activebackground=COLORS['secondary_hover'],
        activeforeground='#ffffff',
        pady=10,
        cursor="hand2",
        relief='flat',
        bd=0,
        command=select_output_folder_model
    ).pack(fill='x')
    
    # Folder info labels
    global model_folder_label, model_folder_info_label, model_output_folder_label, model_output_folder_info
    
    # ===== SIDE-BY-SIDE FOLDER DISPLAY =====
    folders_display_frame = tk.Frame(window_config_card, bg=COLORS['card_bg'])
    folders_display_frame.pack(fill='x', pady=(10, 10))
    
    # LEFT: Input Folder
    input_folder_frame = tk.Frame(folders_display_frame, bg=COLORS['card_bg'])
    input_folder_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
    
    tk.Label(
        input_folder_frame,
        text="Input Folder:",
        font=("Segoe UI", 9, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(fill='x', pady=(0, 2))
    
    model_folder_label = tk.Label(
        input_folder_frame,
        text="No folder selected",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w'
    )
    model_folder_label.pack(fill='x', pady=(0, 2))
    
    model_folder_info_label = tk.Label(
        input_folder_frame,
        text="",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w'
    )
    model_folder_info_label.pack(fill='x', pady=(0, 0))
    
    # RIGHT: Output Folder
    output_folder_frame = tk.Frame(folders_display_frame, bg=COLORS['card_bg'])
    output_folder_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
    
    tk.Label(
        output_folder_frame,
        text="Output Folder:",
        font=("Segoe UI", 9, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(fill='x', pady=(0, 2))
    
    model_output_folder_label = tk.Label(
        output_folder_frame,
        text="Not selected",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w'
    )
    model_output_folder_label.pack(fill='x', pady=(0, 2))
    
    model_output_folder_info = tk.Label(
        output_folder_frame,
        text="",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w'
    )
    model_output_folder_info.pack(fill='x', pady=(0, 0))

    # ===== RIGHT: MODEL PREDICTION (in right column created above) =====
    predict_card_outer, predict_card = _make_card(right_column, "Model Prediction", "🎯")
    predict_card_outer.pack(fill='both', expand=True)

    # ===== Images in folder heading =====
    tk.Label(
        predict_card,
        text="Images in folder:",
        font=("Segoe UI", 11, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(fill='x', pady=(0, 8))
    
    # ===== Image List (with selection for single mode) =====
    listbox_frame = tk.Frame(predict_card, bg=COLORS['card_bg'])
    listbox_frame.pack(fill='x', pady=(0, 10))
    
    scrollbar = tk.Scrollbar(listbox_frame, orient='vertical', width=16)
    scrollbar.pack(side='right', fill='y')
    
    global model_image_listbox
    model_image_listbox = tk.Listbox(
        listbox_frame,
        yscrollcommand=scrollbar.set,
        font=_FONT_BODY,
        bg=COLORS['listbox_bg'],
        fg=COLORS['text'],
        selectbackground=COLORS['listbox_sel'],
        selectforeground='#ffffff',
        activestyle='none',
        relief='flat',
        bd=0,
        highlightthickness=1,
        highlightbackground=COLORS['card_border'],
        height=5  # Fixed height - shows 5 items max
    )
    model_image_listbox.pack(side='left', fill='both', expand=True)
    scrollbar.config(command=model_image_listbox.yview)
    
    # Bind selection event
    model_image_listbox.bind('<<ListboxSelect>>', load_selected_model_image)
    
    # ===== Clear Selection Button (for Batch Mode) =====
    tk.Button(
        predict_card,
        text="🔄  Clear Selection (Batch Mode)",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        activebackground=COLORS['secondary_bg'],
        activeforeground='#ffffff',
        pady=6,
        cursor="hand2",
        relief='flat',
        bd=1,
        command=clear_model_image_selection
    ).pack(fill='x', pady=(0, 15))
    
    # ===== Current Image heading with icon =====
    current_image_header = tk.Frame(predict_card, bg=COLORS['card_bg'])
    current_image_header.pack(fill='x', pady=(0, 8))
    
    tk.Label(
        current_image_header,
        text="📄",
        font=("Segoe UI", 14),
        bg=COLORS['card_bg'],
        fg=COLORS['text']
    ).pack(side='left', padx=(0, 8))
    
    tk.Label(
        current_image_header,
        text="Current Image",
        font=("Segoe UI", 12, "bold"),
        bg=COLORS['card_bg'],
        fg=COLORS['text'],
        anchor='w'
    ).pack(side='left', fill='x', expand=True)
    
    # ===== Current Image Info Label =====
    global model_file_name_label
    model_file_name_label = tk.Label(
        predict_card,
        text="No image selected",
        font=_FONT_BODY,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w',
        wraplength=280,
        justify='left'
    )
    model_file_name_label.pack(fill='x', pady=(0, 2))
    
    global model_file_info_label
    model_file_info_label = tk.Label(
        predict_card,
        text="",
        font=_FONT_SMALL,
        bg=COLORS['card_bg'],
        fg=COLORS['text_muted'],
        anchor='w'
    )
    model_file_info_label.pack(fill='x', pady=(0, 15))
    
    # ===== RUN PREDICTION BUTTON =====
    global _prediction_button
    _prediction_button = tk.Button(
        predict_card,
        text="▶  Run Prediction",
        font=_FONT_BTN,
        bg=COLORS['primary'],
        fg='#ffffff',
        activebackground=COLORS['primary_hover'],
        activeforeground='#ffffff',
        pady=12,
        cursor="hand2",
        relief='flat',
        bd=0,
        highlightthickness=2,
        highlightbackground='#7aa3fc',
        highlightcolor='#7aa3fc',
        command=run_model_prediction_UNIFIED
    )
    _prediction_button.pack(fill='x', pady=(0, 15))

def _on_tab_changed(event):
    """Notebook <<NotebookTabChanged>> — lazy-build model tab on first visit."""
    _build_model_tab()

notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)

# ============================================================
# STATUS BAR (6️⃣ uses set_status helper)
# ============================================================
status_frame = tk.Frame(main_container, bg=COLORS['card_bg'], height=52, bd=0, highlightthickness=0)
status_frame.pack(fill='x', pady=(14, 0))
status_frame.pack_propagate(False)
# top border on status bar
status_top_border = tk.Frame(status_frame, bg=COLORS['card_border'], height=1)
status_top_border.pack(fill='x')

status = tk.Label(
    status_frame,
    text="⚪  Ready – Select input folder to begin",
    font=("Segoe UI", 11, "bold"),
    bg=COLORS['card_bg'],
    fg=COLORS['text_muted'],
    anchor='w',
    padx=20,
    pady=10
)
status.pack(fill='both', expand=True)
_status_widget = status   # 6️⃣ wire up the helper

# ──── Footer ────
footer = tk.Label(
    main_container,
    text="Powered by Rasterio & Tkinter  |  Smart Band Detection  ©  2026",
    font=_FONT_SMALL,
    bg=COLORS['background'],
    fg=COLORS['text_muted']
)
footer.pack(pady=(10, 0))

root.mainloop()