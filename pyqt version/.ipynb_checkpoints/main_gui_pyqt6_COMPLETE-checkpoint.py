#!/usr/bin/env python3
"""
COMPLETE RASTER INDEX CALCULATOR - PyQt6 VERSION (FIXED)
=========================================================
✅ ALL ISSUES FIXED:
1. Image list properly populates
2. All 4 index options in dropdown
3. "All Indices" button added
4. Complete integration with fixed utils
"""

import sys
import os
import platform
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QListWidget, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QTabWidget, QFrame,
    QScrollArea, QGroupBox, QSpinBox, QDoubleSpinBox, QSlider,
    QListWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QCursor

# Scientific computing
import numpy as np
import rasterio
from PIL import Image as PILImage

# ML libraries
import tensorflow as tf

# Import our FIXED utility modules
from utils_core_FIXED import (
    dice_loss, dice_coef, iou_score, CUSTOM_OBJECTS,
    preprocess_for_model,
    detect_band_by_type, cached_detect_band, get_required_bands,
    calculate_ndsi, calculate_ndvi, calculate_ndwi,
    load_composite_image_8_bands, check_composite_exists,
    save_geotiff_probability_only, save_ndsi_geotiff,
    generate_prediction_report, validate_model_with_composite,
    calculate_all_indices_for_image  # NEW: Critical function!
)

from utils_model_FIXED import (
    predict_with_sliding_window,
    predict_single_image,
    predict_batch_images,
    load_keras_model
)

# ============================================================
# COLOR SCHEME - Professional Dark Theme
# ============================================================

COLORS = {
    'background': '#0a0e27',
    'card_bg': '#151b3d',
    'card_border': '#2a3254',
    'primary': '#5b8def',
    'primary_hover': '#4a7dde',
    'primary_disabled': '#3a5a9e',
    'secondary_bg': '#1e2749',
    'secondary_hover': '#2a3461',
    'text': '#e8e9f0',
    'text_muted': '#8b92b8',
    'success': '#5dca88',
    'warning': '#f59e42',
    'error': '#ef5b5b',
    'listbox_bg': '#0f1229',
    'listbox_sel': '#4a7dde',
    'details_bg': '#0c0f1f'
}

# ============================================================
# WORKER THREADS FOR BACKGROUND PROCESSING
# ============================================================

class NDSICalculationWorker(QThread):
    """Worker thread for NDSI/NDVI calculations"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)
    
    def __init__(self, folder_path, output_folder, index_type, threshold):
        super().__init__()
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.index_type = index_type
        self.threshold = threshold
        self.is_cancelled = False
        
    def run(self):
        try:
            # Find all TIFF files
            files = [f for f in os.listdir(self.folder_path) 
                    if f.lower().endswith(('.tif', '.tiff'))]
            
            if not files:
                self.finished.emit(False, "No TIFF files found")
                return
            
            total = len(files)
            success_count = 0
            
            for idx, filename in enumerate(files):
                if self.is_cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return
                
                self.progress.emit(idx + 1, total, f"Processing {filename}")
                
                try:
                    filepath = os.path.join(self.folder_path, filename)
                    
                    # Open image and detect bands
                    with rasterio.open(filepath) as src:
                        band_result = get_required_bands(src, self.index_type)
                        
                        if not band_result['success']:
                            self.log.emit(f"⚠️ Skipping {filename}: {band_result['info']}")
                            continue
                        
                        # Calculate index
                        if self.index_type == 'NDSI':
                            index_data = calculate_ndsi(
                                band_result['bands']['GREEN']['data'],
                                band_result['bands']['SWIR']['data']
                            )
                        elif self.index_type == 'NDVI':
                            index_data = calculate_ndvi(
                                band_result['bands']['RED']['data'],
                                band_result['bands']['NIR']['data']
                            )
                        else:
                            continue
                        
                        # Save result
                        base_name = os.path.splitext(filename)[0]
                        output_path = os.path.join(
                            self.output_folder,
                            f"{base_name}_{self.index_type.lower()}_result.tif"
                        )
                        
                        save_ndsi_geotiff(index_data, filepath, output_path, self.threshold)
                        
                        self.log.emit(f"✅ Processed: {filename}")
                        success_count += 1
                        
                except Exception as e:
                    self.log.emit(f"❌ Error processing {filename}: {e}")
                    continue
            
            self.finished.emit(True, f"Completed: {success_count}/{total} files processed")
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def cancel(self):
        self.is_cancelled = True


class AllIndicesWorker(QThread):
    """Worker thread for calculating all indices - supports batch mode"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str, dict)  # success, message, results
    log = pyqtSignal(str)
    
    def __init__(self, image_paths, output_folder):
        super().__init__()
        # Can be single path (string) or multiple paths (list)
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
        else:
            self.image_paths = image_paths
        self.output_folder = output_folder
        
    def run(self):
        try:
            total = len(self.image_paths)
            success_count = 0
            results = []
            
            for idx, image_path in enumerate(self.image_paths):
                filename = os.path.basename(image_path)
                self.progress.emit(idx + 1, total, f"Processing {filename}")
                self.log.emit(f"\n🔍 Processing: {filename}")
                
                result = calculate_all_indices_for_image(image_path, self.output_folder)
                
                if result['success']:
                    success_count += 1
                    self.log.emit(f"✅ Created: {os.path.basename(result['composite_path'])}")
                    results.append(result)
                else:
                    self.log.emit(f"❌ Failed: {result['error']}")
            
            if success_count > 0:
                self.finished.emit(
                    True, 
                    f"Completed: {success_count}/{total} composites created",
                    {'count': success_count, 'total': total, 'results': results}
                )
            else:
                self.finished.emit(False, "All images failed", {})
                
        except Exception as e:
            self.finished.emit(False, str(e), {})


class ModelPredictionWorker(QThread):
    """Worker thread for model predictions"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str, dict)  # success, message, results
    log = pyqtSignal(str)
    
    # ✅ NEW: Dual progress signals
    overall_progress = pyqtSignal(int)  # Overall batch progress (0-100)
    image_progress = pyqtSignal(int, str)  # Individual image progress (0-100), image name
    
    def __init__(self, model, image_path, window_size, stride, output_folder, mode='single'):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.window_size = window_size
        self.stride = stride
        self.output_folder = output_folder
        self.mode = mode  # 'single' or 'batch'
        self.is_cancelled = False
        
    def run(self):
        try:
            if self.mode == 'single':
                self.log.emit("🚀 Starting single image prediction...")
                
                # ✅ Initialize progress bars
                self.overall_progress.emit(0)
                self.image_progress.emit(0, os.path.basename(self.image_path))
                
                # Define callbacks for GUI updates
                def log_callback(msg):
                    self.log.emit(msg)
                
                # ✅ Window progress callback for real-time updates
                def window_progress_callback(current, total, pct):
                    """Called from predict_with_sliding_window for each progress update"""
                    self.image_progress.emit(pct, os.path.basename(self.image_path))
                    self.overall_progress.emit(pct)
                
                result = predict_single_image(
                    self.model,
                    self.image_path,
                    self.window_size,
                    self.stride,
                    self.output_folder,
                    progress_callback=window_progress_callback,
                    log_callback=log_callback
                )
                
                if result['success']:
                    # ✅ Complete progress bars
                    self.overall_progress.emit(100)
                    self.image_progress.emit(100, os.path.basename(self.image_path))
                    self.finished.emit(True, "Prediction complete!", result)
                else:
                    self.finished.emit(False, result['error'], {})
                    
            else:  # batch mode
                self.log.emit("📦 Starting batch prediction...")
                
                # ✅ Initialize progress
                self.overall_progress.emit(0)
                
                def log_callback(msg):
                    self.log.emit(msg)
                
                # ✅ Get image list for batch tracking
                folder = os.path.dirname(self.image_path)
                files = [f for f in os.listdir(folder) 
                        if f.lower().endswith(('.tif', '.tiff')) and '_composite' in f.lower()]
                total_images = len(files)
                
                self.log.emit(f"📋 Found {total_images} images to process")
                
                # Process each image with progress tracking
                success_count = 0
                for idx, filename in enumerate(files):
                    image_path = os.path.join(folder, filename)
                    
                    self.log.emit(f"\n📄 [{idx+1}/{total_images}] Processing: {filename}")
                    
                    # ✅ Reset image progress for new image
                    self.image_progress.emit(0, filename)
                    
                    # ✅ Update overall progress based on completed images
                    overall_base = int((idx / total_images) * 100)
                    self.overall_progress.emit(overall_base)
                    
                    # ✅ Window callback for this specific image
                    def make_window_callback(image_idx, img_name, total_imgs):
                        """Factory function to avoid closure issues"""
                        def callback(current, total, pct):
                            # Update individual image progress
                            self.image_progress.emit(pct, img_name)
                            
                            # Calculate overall progress
                            completed_pct = (image_idx / total_imgs) * 100
                            current_image_contribution = (pct / 100) * (100 / total_imgs)
                            total_pct = int(completed_pct + current_image_contribution)
                            self.overall_progress.emit(total_pct)
                        return callback
                    
                    window_callback = make_window_callback(idx, filename, total_images)
                    
                    result = predict_single_image(
                        self.model,
                        image_path,
                        self.window_size,
                        self.stride,
                        self.output_folder,
                        progress_callback=window_callback,
                        log_callback=log_callback
                    )
                    
                    if result['success']:
                        success_count += 1
                        self.log.emit(f"   ✅ Success: {filename}")
                    else:
                        self.log.emit(f"   ❌ Failed: {filename}")
                    
                    # ✅ Mark this image as 100% complete
                    self.image_progress.emit(100, filename)
                    
                    # ✅ Update overall progress after image completion
                    completed_overall = int(((idx + 1) / total_images) * 100)
                    self.overall_progress.emit(completed_overall)
                
                # ✅ Complete overall progress
                self.overall_progress.emit(100)
                
                if success_count > 0:
                    self.finished.emit(True, f"Batch complete: {success_count}/{total_images}", 
                                     {'successful': success_count, 'total': total_images})
                else:
                    self.finished.emit(False, "All predictions failed", {})
                    
        except Exception as e:
            self.finished.emit(False, str(e), {})
    
    def cancel(self):
        self.is_cancelled = True


# ============================================================
# MAIN WINDOW
# ============================================================

class RasterCalculatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌍 Raster Index Calculator - PyQt6")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.folder_path = None
        self.output_folder_path = None
        self.current_image_path = None
        self.image_files = []
        self.loaded_model = None
        self.model_info = None
        self.model_folder_path = None
        self.model_output_folder = None
        self.model_input_image_path = None
        self.window_size = 128
        self.stride = 64
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
        # Show welcome message
        self.update_status("⚪ Ready – Select input folder to begin", "info")
    
    def setup_ui(self):
        """Setup the main UI"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # Create tabs
        self.ndsi_tab = self.create_ndsi_tab()
        self.model_tab = self.create_model_tab()
        
        self.tabs.addTab(self.ndsi_tab, "📊 NDSI / NDVI Calculation")
        self.tabs.addTab(self.model_tab, "🤖 ML Model Prediction")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        self.status_label = QLabel("⚪ Ready")
        self.status_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(status_widget)
        
        # Footer
        footer = QLabel("Powered by Rasterio & PyQt6  |  Smart Band Detection  ©  2026")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)
    
    def create_header(self):
        """Create header with title"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("🌍  Raster Index Calculator")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        
        subtitle = QLabel("Professional GeoTIFF Analysis Tool with Smart Band Detection")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 11))
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        
        return header_widget
    
    def create_ndsi_tab(self):
        """Create NDSI/NDVI calculation tab - FIXED VERSION"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(20)
        
        # Left column - Controls
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        
        # Folders section
        folders_group = self.create_group_box("📁 Folders")
        folders_layout = QVBoxLayout()
        
        # Input folder button
        input_btn = QPushButton("📂  Select Input Folder")
        input_btn.clicked.connect(self.select_input_folder)
        folders_layout.addWidget(input_btn)
        
        # Output folder button
        output_btn = QPushButton("💾  Select Output Folder")
        output_btn.clicked.connect(self.select_output_folder_ndsi)
        folders_layout.addWidget(output_btn)
        
        # Folder info labels
        self.input_folder_label = QLabel("No folder selected")
        self.output_folder_label = QLabel("Not selected")
        folders_layout.addWidget(QLabel("Input Folder:"))
        folders_layout.addWidget(self.input_folder_label)
        folders_layout.addWidget(QLabel("Output Folder:"))
        folders_layout.addWidget(self.output_folder_label)
        
        folders_group.setLayout(folders_layout)
        left_layout.addWidget(folders_group)
        
        # ===== NEW: IMAGE LIST SECTION =====
        images_group = self.create_group_box("🖼️ Images in Folder")
        images_layout = QVBoxLayout()
        
        # Image listbox
        self.image_listbox = QListWidget()
        self.image_listbox.itemClicked.connect(self.load_selected_image)
        images_layout.addWidget(self.image_listbox)
        
        # Clear selection button
        clear_btn = QPushButton("🔄 Clear Selection")
        clear_btn.clicked.connect(self.clear_image_selection)
        images_layout.addWidget(clear_btn)
        
        images_group.setLayout(images_layout)
        left_layout.addWidget(images_group)
        
        # Current Image section
        current_image_group = self.create_group_box("📄 Current Image")
        current_image_layout = QVBoxLayout()
        
        self.file_name_label = QLabel("No image selected")
        self.file_info_label = QLabel("")
        current_image_layout.addWidget(self.file_name_label)
        current_image_layout.addWidget(self.file_info_label)
        
        current_image_group.setLayout(current_image_layout)
        left_layout.addWidget(current_image_group)
        
        # Calculation section - FIXED: ALL 4 OPTIONS
        calc_group = self.create_group_box("⚙️ Calculation")
        calc_layout = QVBoxLayout()
        
        # Index type - FIXED: Added all 4 options
        calc_layout.addWidget(QLabel("Index Type:"))
        self.index_type_combo = QComboBox()
        self.index_type_combo.addItems([
            "NDSI (Snow)",
            "NDVI (Vegetation)",
            "NDWI (Water)",
            "All Indices (8-band: B2,B3,B4,B8,B11,NDSI,NDVI,NDWI)"
        ])
        calc_layout.addWidget(self.index_type_combo)
        
        # Threshold
        calc_layout.addWidget(QLabel("NDSI Threshold:"))
        self.ndsi_threshold_spin = QDoubleSpinBox()
        self.ndsi_threshold_spin.setRange(-1.0, 1.0)
        self.ndsi_threshold_spin.setSingleStep(0.05)
        self.ndsi_threshold_spin.setValue(0.40)
        self.ndsi_threshold_spin.setDecimals(2)
        calc_layout.addWidget(self.ndsi_threshold_spin)
        
        calc_group.setLayout(calc_layout)
        left_layout.addWidget(calc_group)
        
        # Actions
        actions_group = self.create_group_box("🚀 Actions")
        actions_layout = QVBoxLayout()
        
        self.run_calculation_btn = QPushButton("▶  Run Calculation")
        self.run_calculation_btn.clicked.connect(self.run_calculation)
        actions_layout.addWidget(self.run_calculation_btn)
        
        actions_group.setLayout(actions_layout)
        left_layout.addWidget(actions_group)
        
        left_layout.addStretch()
        
        # Right column - Preview
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        
        preview_group = self.create_group_box("🖼️ Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel("No image loaded\n📂 Select an image to see preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("border: 2px dashed #2a3254;")
        
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)
        
        # Log output
        log_group = self.create_group_box("📝 Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Add columns
        layout.addWidget(left_column, 1)
        layout.addWidget(right_column, 2)
        
        return tab
    
    def create_model_tab(self):
        """Create Model Loading and Prediction tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(20)
        
        # Left column
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        
        # Model section
        model_group = self.create_group_box("🤖 Machine Learning Model")
        model_layout = QVBoxLayout()
        
        upload_btn = QPushButton("📤  Upload Model (.keras/.h5)")
        upload_btn.clicked.connect(self.upload_model)
        model_layout.addWidget(upload_btn)
        
        self.model_details_btn = QPushButton("📊  Model Details")
        self.model_details_btn.setEnabled(False)
        self.model_details_btn.clicked.connect(self.show_model_details)
        model_layout.addWidget(self.model_details_btn)
        
        self.model_info_label = QLabel("No model loaded")
        model_layout.addWidget(self.model_info_label)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Window settings
        window_group = self.create_group_box("🔲 Sliding Window Settings")
        window_layout = QVBoxLayout()
        
        window_layout.addWidget(QLabel("Window Size:"))
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(32, 512)
        self.window_size_spin.setSingleStep(32)
        self.window_size_spin.setValue(128)
        self.window_size_spin.valueChanged.connect(self.on_window_size_changed)
        window_layout.addWidget(self.window_size_spin)
        
        window_layout.addWidget(QLabel("Stride:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(8, 256)
        self.stride_spin.setSingleStep(8)
        self.stride_spin.setValue(64)
        window_layout.addWidget(self.stride_spin)
        
        apply_btn = QPushButton("✓  Apply Settings")
        apply_btn.clicked.connect(self.apply_window_settings)
        window_layout.addWidget(apply_btn)
        
        window_group.setLayout(window_layout)
        left_layout.addWidget(window_group)
        
        # Folders
        folders_group = self.create_group_box("📁 Model Prediction")
        folders_layout = QVBoxLayout()
        
        input_btn = QPushButton("📂  Select Input Folder")
        input_btn.clicked.connect(self.select_model_input_folder)
        folders_layout.addWidget(input_btn)
        
        output_btn = QPushButton("💾  Select Output Folder")
        output_btn.clicked.connect(self.select_model_output_folder)
        folders_layout.addWidget(output_btn)
        
        self.model_input_label = QLabel("No folder selected")
        self.model_output_label = QLabel("Not selected")
        folders_layout.addWidget(QLabel("Input Folder:"))
        folders_layout.addWidget(self.model_input_label)
        folders_layout.addWidget(QLabel("Output Folder:"))
        folders_layout.addWidget(self.model_output_label)
        
        folders_group.setLayout(folders_layout)
        left_layout.addWidget(folders_group)
        
        left_layout.addStretch()
        
        # Right column
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        
        images_group = self.create_group_box("🎯 Model Prediction")
        images_layout = QVBoxLayout()
        
        images_layout.addWidget(QLabel("Images in folder:"))
        self.model_image_listbox = QListWidget()
        self.model_image_listbox.itemSelectionChanged.connect(self.load_selected_model_image)
        images_layout.addWidget(self.model_image_listbox)
        
        clear_btn = QPushButton("🔄  Clear Selection (Batch Mode)")
        clear_btn.clicked.connect(self.clear_model_selection)
        images_layout.addWidget(clear_btn)
        
        images_layout.addWidget(QLabel("📄 Current Image"))
        self.model_file_name_label = QLabel("No image selected")
        self.model_file_info_label = QLabel("")
        images_layout.addWidget(self.model_file_name_label)
        images_layout.addWidget(self.model_file_info_label)
        
        self.run_prediction_btn = QPushButton("▶  Run Prediction")
        self.run_prediction_btn.clicked.connect(self.run_model_prediction)
        self.run_prediction_btn.setEnabled(False)
        images_layout.addWidget(self.run_prediction_btn)
        
        images_group.setLayout(images_layout)
        right_layout.addWidget(images_group)
        
        # ✅ NEW: Dual Progress Bars
        progress_group = self.create_group_box("📊 Progress")
        progress_layout = QVBoxLayout()
        
        progress_layout.addWidget(QLabel("🌍 Overall Progress:"))
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.overall_progress_bar)
        
        self.current_image_label = QLabel("📄 Current Image: -")
        progress_layout.addWidget(self.current_image_label)
        
        self.image_progress_bar = QProgressBar()
        self.image_progress_bar.setRange(0, 100)
        self.image_progress_bar.setValue(0)
        self.image_progress_bar.setTextVisible(True)
        self.image_progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.image_progress_bar)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # Log
        log_group = self.create_group_box("📝 Prediction Log")
        log_layout = QVBoxLayout()
        
        self.model_log_text = QTextEdit()
        self.model_log_text.setReadOnly(True)
        log_layout.addWidget(self.model_log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        layout.addWidget(left_column, 1)
        layout.addWidget(right_column, 1)
        
        return tab
    
    def create_group_box(self, title):
        """Create styled group box"""
        group = QGroupBox(title)
        return group
    
    def apply_styles(self):
        """Apply dark theme stylesheet"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QWidget {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
            }}
            QGroupBox {{
                background-color: {COLORS['card_bg']};
                border: 1px solid {COLORS['card_border']};
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                font-weight: bold;
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['primary_disabled']};
            }}
            QLabel {{
                color: {COLORS['text']};
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS['secondary_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['card_border']};
                border-radius: 4px;
                padding: 8px;
            }}
            QListWidget {{
                background-color: {COLORS['listbox_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['card_border']};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['listbox_sel']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['card_border']};
                background: {COLORS['card_bg']};
            }}
            QTabBar::tab {{
                background: {COLORS['secondary_bg']};
                color: {COLORS['text']};
                padding: 10px 20px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
            }}
            QTextEdit {{
                background-color: {COLORS['listbox_bg']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['card_border']};
                border-radius: 4px;
            }}
            QProgressBar {{
                border: 1px solid {COLORS['card_border']};
                border-radius: 4px;
                text-align: center;
                background: {COLORS['secondary_bg']};
                height: 25px;
                font-size: 11pt;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
    
    # ========================================
    # FOLDER SELECTION METHODS - FIXED
    # ========================================
    
    def select_input_folder(self):
        """Select input folder for NDSI calculation - FIXED"""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_path = folder
            self.input_folder_label.setText(os.path.basename(folder))
            self.update_status(f"✅ Input folder selected", "success")
            
            # FIXED: Load and display images in listbox
            self.image_listbox.clear()
            self.image_files = [f for f in os.listdir(folder) 
                              if f.lower().endswith(('.tif', '.tiff'))]
            self.image_files.sort()
            
            for file in self.image_files:
                self.image_listbox.addItem(file)
            
            self.log_text.append(f"✅ Found {len(self.image_files)} images in folder")
    
    def select_output_folder_ndsi(self):
        """Select output folder for NDSI results"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_path = folder
            self.output_folder_label.setText(os.path.basename(folder))
            self.update_status(f"✅ Output folder selected", "success")
    
    def load_selected_image(self, item):
        """Load selected image from listbox - NEW"""
        if item:
            filename = item.text()
            self.current_image_path = os.path.join(self.folder_path, filename)
            self.file_name_label.setText(f"Selected: {filename}")
            
            # Try to load preview
            try:
                with rasterio.open(self.current_image_path) as src:
                    info = f"{src.width}x{src.height} | {src.count} bands"
                    self.file_info_label.setText(info)
                    self.log_text.append(f"📂 Loaded: {filename}")
            except Exception as e:
                self.file_info_label.setText("Could not read file info")
    
    def clear_image_selection(self):
        """Clear image selection - NEW"""
        self.image_listbox.clearSelection()
        self.current_image_path = None
        self.file_name_label.setText("No image selected")
        self.file_info_label.setText("")
    
    def select_model_input_folder(self):
        """Select input folder for model prediction - FIXED"""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Input Folder")
        if folder:
            self.model_folder_path = folder
            self.model_input_label.setText(os.path.basename(folder))
            self.load_model_images()
            self.update_status(f"✅ Model input folder selected", "success")
    
    def select_model_output_folder(self):
        """Select output folder for predictions"""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Output Folder")
        if folder:
            self.model_output_folder = folder
            self.model_output_label.setText(os.path.basename(folder))
            self.update_status(f"✅ Model output folder selected", "success")
    
    # ========================================
    # MODEL METHODS
    # ========================================
    
    def upload_model(self):
        """Upload and load ML model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Keras Models (*.keras *.h5)"
        )
        
        if file_path:
            try:
                self.update_status("⏳ Loading model...", "warning")
                self.model_log_text.append("📦 Loading model...")
                
                # Use fixed load function
                self.loaded_model = load_keras_model(file_path, self.model_log_text.append)
                
                if self.loaded_model is None:
                    QMessageBox.critical(self, "Error", "Failed to load model")
                    return
                
                # Get model info
                patch_size = self.loaded_model.input_shape[1]
                self.window_size = patch_size
                self.stride = max(1, patch_size // 2)
                
                self.window_size_spin.setValue(self.window_size)
                self.stride_spin.setValue(self.stride)
                
                # Update UI
                self.model_info_label.setText(
                    f"✅ Model loaded\n"
                    f"Input: {self.loaded_model.input_shape}\n"
                    f"Window: {patch_size}px"
                )
                self.model_details_btn.setEnabled(True)
                self.run_prediction_btn.setEnabled(True)
                
                self.update_status("✅ Model loaded successfully!", "success")
                self.model_log_text.append("✅ Model loaded successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
                self.update_status("❌ Model loading failed", "error")
                self.model_log_text.append(f"❌ Error: {str(e)}")
    
    def show_model_details(self):
        """Show detailed model information"""
        if self.loaded_model:
            msg = QMessageBox()
            msg.setWindowTitle("Model Details")
            msg.setText(
                f"Input Shape: {self.loaded_model.input_shape}\n"
                f"Output Shape: {self.loaded_model.output_shape}\n"
                f"Total Parameters: {self.loaded_model.count_params():,}"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
    
    def load_model_images(self):
        """Load images from model input folder - FIXED"""
        if not self.model_folder_path:
            return
        
        self.model_image_listbox.clear()
        
        files = [f for f in os.listdir(self.model_folder_path) 
                if f.lower().endswith(('.tif', '.tiff'))]
        files.sort()
        
        for file in files:
            self.model_image_listbox.addItem(file)
        
        self.update_status(f"✅ Found {len(files)} images", "success")
        self.model_log_text.append(f"✅ Found {len(files)} images in folder")
    
    def load_selected_model_image(self):
        """Load selected image for preview"""
        selected_items = self.model_image_listbox.selectedItems()
        if selected_items:
            filename = selected_items[0].text()
            self.model_input_image_path = os.path.join(self.model_folder_path, filename)
            self.model_file_name_label.setText(f"Selected: {filename}")
            
            # Check if composite exists
            has_composite = check_composite_exists(self.model_input_image_path)
            if has_composite:
                self.model_file_info_label.setText("✅ Composite found")
            else:
                self.model_file_info_label.setText("⚠️ No composite - run All Indices first")
    
    def clear_model_selection(self):
        """Clear image selection for batch mode"""
        self.model_image_listbox.clearSelection()
        self.model_input_image_path = None
        self.model_file_name_label.setText("Batch Mode - All images")
        self.model_file_info_label.setText("")
    
    def on_window_size_changed(self, value):
        """Auto-adjust stride when window size changes"""
        suggested_stride = max(1, value // 2)
        self.stride_spin.setValue(suggested_stride)
    
    def apply_window_settings(self):
        """Apply window settings"""
        self.window_size = self.window_size_spin.value()
        self.stride = self.stride_spin.value()
        self.update_status(f"✅ Settings applied: Window={self.window_size}, Stride={self.stride}", "success")
    
    # ========================================
    # CALCULATION METHODS - FIXED
    # ========================================
    
    def run_calculation(self):
        """Run selected calculation - FIXED WITH ALL INDICES SUPPORT"""
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Please select input folder first!")
            return
        
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "Please select output folder!")
            return
        
        # Get selected index
        index_text = self.index_type_combo.currentText()
        
        # FIXED: Handle "All Indices" option
        if "All Indices" in index_text:
            self.run_all_indices_calculation()
            return
        
        # Extract index type
        if "NDSI" in index_text:
            index_type = "NDSI"
        elif "NDVI" in index_text:
            index_type = "NDVI"
        elif "NDWI" in index_text:
            index_type = "NDWI"
        else:
            QMessageBox.warning(self, "Error", "Unknown index type!")
            return
        
        # Get threshold
        threshold = self.ndsi_threshold_spin.value()
        
        # Start worker thread
        self.log_text.append(f"\n🚀 Starting {index_type} calculation...")
        self.update_status(f"⏳ Calculating {index_type}...", "warning")
        
        self.worker = NDSICalculationWorker(
            self.folder_path,
            self.output_folder_path,
            index_type,
            threshold
        )
        
        self.worker.progress.connect(self.on_calculation_progress)
        self.worker.log.connect(self.log_text.append)
        self.worker.finished.connect(self.on_calculation_finished)
        
        self.worker.start()
        self.run_calculation_btn.setEnabled(False)
    
    def run_all_indices_calculation(self):
        """Run ALL INDICES calculation - supports batch mode"""
        if not self.current_image_path and not self.image_files:
            QMessageBox.warning(
                self, 
                "No Images", 
                "No images available!\n\nPlease select input folder first."
            )
            return
        
        # Determine mode: single or batch
        if self.current_image_path:
            # Single image mode
            image_paths = [self.current_image_path]
            mode_msg = f"Processing 1 image"
        else:
            # Batch mode - all images in folder
            image_paths = [os.path.join(self.folder_path, f) for f in self.image_files]
            mode_msg = f"Processing {len(image_paths)} images (BATCH MODE)"
        
        self.log_text.append(f"\n🎨 Starting ALL INDICES calculation...")
        self.log_text.append(f"   {mode_msg}")
        self.update_status("⏳ Calculating all indices...", "warning")
        
        # Start worker
        self.all_indices_worker = AllIndicesWorker(
            image_paths,
            self.output_folder_path
        )
        
        self.all_indices_worker.progress.connect(self.on_all_indices_progress)
        self.all_indices_worker.log.connect(self.log_text.append)
        self.all_indices_worker.finished.connect(self.on_all_indices_finished)
        
        self.all_indices_worker.start()
        self.run_calculation_btn.setEnabled(False)
    
    def on_all_indices_progress(self, current, total, message):
        """Update progress for all indices"""
        self.update_status(f"⏳ Processing {current}/{total}: {message}", "warning")
    
    def on_all_indices_finished(self, success, message, result):
        """Handle all indices completion - supports batch mode"""
        self.run_calculation_btn.setEnabled(True)
        
        if success:
            count = result.get('count', 1)
            total = result.get('total', 1)
            
            self.update_status(f"✅ Created {count} composite(s)!", "success")
            self.log_text.append(f"\n✅ SUCCESS: {count}/{total} composites created!")
            
            # Build message
            if count == 1:
                # Single image
                composite_path = result['results'][0]['composite_path']
                msg = (
                    f"✅ 8-band composite created!\n\n"
                    f"File:\n"
                    f"• {os.path.basename(composite_path)}\n\n"
                    f"Contains 8 bands:\n"
                    f"  B2, B3, B4, B8, B11, NDSI, NDVI, NDWI\n\n"
                    f"Location: {self.output_folder_path}"
                )
            else:
                # Batch mode
                msg = (
                    f"✅ Batch processing complete!\n\n"
                    f"Created: {count}/{total} composites\n\n"
                    f"Each contains 8 bands:\n"
                    f"  B2, B3, B4, B8, B11, NDSI, NDVI, NDWI\n\n"
                    f"Location: {self.output_folder_path}"
                )
            
            QMessageBox.information(self, "Success!", msg)
        else:
            self.update_status("❌ Processing failed", "error")
            self.log_text.append(f"\n❌ Error: {message}")
            QMessageBox.critical(self, "Error", message)
    
    def on_calculation_progress(self, current, total, message):
        """Update progress"""
        self.update_status(f"⏳ Processing {current}/{total}: {message}", "warning")
    
    def on_calculation_finished(self, success, message):
        """Handle calculation completion"""
        self.run_calculation_btn.setEnabled(True)
        
        if success:
            self.update_status(f"✅ {message}", "success")
            self.log_text.append(f"\n✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.update_status(f"❌ {message}", "error")
            self.log_text.append(f"\n❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    # ========================================
    # MODEL PREDICTION METHODS
    # ========================================
    
    def run_model_prediction(self):
        """Run model prediction"""
        if not self.loaded_model:
            QMessageBox.warning(self, "Error", "Please load a model first!")
            return
        
        if not self.model_folder_path:
            QMessageBox.warning(self, "Error", "Please select input folder!")
            return
        
        if not self.model_output_folder:
            QMessageBox.warning(self, "Error", "Please select output folder!")
            return
        
        # ✅ Reset progress bars
        self.overall_progress_bar.setValue(0)
        self.image_progress_bar.setValue(0)
        self.current_image_label.setText("📄 Current Image: -")
        
        # Determine mode
        if self.model_input_image_path:
            mode = 'single'
            image_path = self.model_input_image_path
        else:
            mode = 'batch'
            # Use first image in folder for batch mode
            files = [f for f in os.listdir(self.model_folder_path) 
                    if f.lower().endswith(('.tif', '.tiff'))]
            if not files:
                QMessageBox.warning(self, "Error", "No images in folder!")
                return
            image_path = os.path.join(self.model_folder_path, files[0])
        
        # Start prediction
        self.model_log_text.append(f"\n🚀 Starting {mode} prediction...")
        self.update_status(f"⏳ Running prediction ({mode})...", "warning")
        
        self.pred_worker = ModelPredictionWorker(
            self.loaded_model,
            image_path,
            self.window_size,
            self.stride,
            self.model_output_folder,
            mode
        )
        
        # ✅ Connect progress bar signals
        self.pred_worker.overall_progress.connect(self.on_overall_progress)
        self.pred_worker.image_progress.connect(self.on_image_progress)
        
        self.pred_worker.progress.connect(self.on_prediction_progress)
        self.pred_worker.log.connect(self.model_log_text.append)
        self.pred_worker.finished.connect(self.on_prediction_finished)
        
        self.pred_worker.start()
        self.run_prediction_btn.setEnabled(False)
    
    
    # ✅ SAFE: Progress bar update methods (NO repaint calls!)
    def on_overall_progress(self, percent):
        """Update overall/batch progress bar"""
        self.overall_progress_bar.setValue(percent)
    
    def on_image_progress(self, percent, image_name):
        """Update individual image progress bar"""
        self.image_progress_bar.setValue(percent)
        self.current_image_label.setText(f"📄 Current Image: {image_name}")
    
    def on_prediction_progress(self, current, total, message):
        """Update prediction progress"""
        self.update_status(f"⏳ Predicting {current}/{total}: {message}", "warning")
    
    def on_prediction_finished(self, success, message, result):
        """Handle prediction completion"""
        self.run_prediction_btn.setEnabled(True)
        
        if success:
            # ✅ Set progress bars to 100% on success
            self.overall_progress_bar.setValue(100)
            self.image_progress_bar.setValue(100)
            self.update_status(f"✅ {message}", "success")
            self.model_log_text.append(f"\n✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.update_status(f"❌ {message}", "error")
            self.model_log_text.append(f"\n❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def update_status(self, text, level="info"):
        """Update status bar"""
        colors = {
            'info': COLORS['text_muted'],
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'error': COLORS['error']
        }
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {colors.get(level, COLORS['text'])};")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = RasterCalculatorWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()