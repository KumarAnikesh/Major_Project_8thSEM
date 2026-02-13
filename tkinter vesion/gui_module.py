"""
===============================================================================
GUI MODULE - Raster Index Calculator
===============================================================================
Main GUI module that handles all UI components and coordinates between
the image processing and model prediction modules.
===============================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image as PILImage, ImageTk
import os
import platform

# Import our custom modules
from image_processing_module import ImageProcessor
from model_prediction_module import ModelPredictor


# ============================================================
# COLOR SCHEME & FONTS
# ============================================================
COLORS = {
    'background': '#1a1d24',
    'card_bg': '#22262e',
    'card_border': '#363b46',
    'primary': '#5b8def',
    'primary_hover': '#7aa3fc',
    'secondary_bg': '#2c313a',
    'text': '#e8eaed',
    'text_muted': '#9ba1ab',
    'listbox_bg': '#2a2e36',
    'listbox_sel': '#5b8def',
    'tab_sel': '#5b8def',
    'tab_unsel': '#2a2e36',
    'tab_border': '#363b46'
}

_FONT_TITLE = ("Segoe UI", 16, "bold")
_FONT_HEADING = ("Segoe UI", 12, "bold")
_FONT_BODY = ("Segoe UI", 10)
_FONT_SMALL = ("Segoe UI", 9)
_FONT_BTN = ("Segoe UI", 11, "bold")


class RasterCalculatorGUI:
    """Main GUI class for Raster Index Calculator"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = tk.Tk()
        self.root.title("Raster Index Calculator – Professional Edition")
        
        # Initialize modules
        self.image_processor = ImageProcessor()
        self.model_predictor = ModelPredictor()
        
        # GUI state variables
        self.folder_path = None
        self.output_folder_path = None
        self.model_folder_path = None
        self.model_output_folder = None
        self.loaded_model = None
        self.model_input_size = None
        self.window_size_var = None
        self.stride_var = None
        self.current_image = None
        self.current_image_data = None
        self.model_batch_mode = True
        
        # Setup GUI
        self._setup_window()
        self._create_widgets()
        
    def _setup_window(self):
        """Configure main window properties"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size (90% of screen)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=COLORS['background'])
        
        # Platform-specific settings
        if platform.system() == "Darwin":  # macOS
            try:
                self.root.createcommand('tk::mac::Quit', self.root.quit)
            except:
                pass
                
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=COLORS['background'], padx=30, pady=20)
        main_container.pack(fill='both', expand=True)
        
        # Header
        self._create_header(main_container)
        
        # Notebook (tabs)
        self._create_notebook(main_container)
        
        # Status bar
        self._create_status_bar(main_container)
        
        # Footer
        self._create_footer(main_container)
        
    def _create_header(self, parent):
        """Create application header"""
        header = tk.Frame(parent, bg=COLORS['background'])
        header.pack(fill='x', pady=(0, 20))
        
        # Icon + Title
        title_frame = tk.Frame(header, bg=COLORS['background'])
        title_frame.pack()
        
        tk.Label(
            title_frame,
            text="🗺️",
            font=("Segoe UI", 32),
            bg=COLORS['background'],
            fg=COLORS['text']
        ).pack(side='left', padx=(0, 15))
        
        tk.Label(
            title_frame,
            text="Raster Index Calculator",
            font=("Segoe UI", 24, "bold"),
            bg=COLORS['background'],
            fg=COLORS['text']
        ).pack(side='left')
        
        # Subtitle
        tk.Label(
            header,
            text="Professional GeoTIFF Analysis Tool with Smart Band Detection",
            font=_FONT_BODY,
            bg=COLORS['background'],
            fg=COLORS['text_muted']
        ).pack(pady=(5, 0))
        
    def _create_notebook(self, parent):
        """Create tabbed interface"""
        # Custom style for tabs
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure(
            'Custom.TNotebook',
            background=COLORS['background'],
            borderwidth=0,
            tabmargins=[0, 10, 0, 0]
        )
        
        style.configure(
            'Custom.TNotebook.Tab',
            background=COLORS['tab_unsel'],
            foreground=COLORS['text'],
            padding=[20, 12],
            borderwidth=0,
            font=("Segoe UI", 11, "bold")
        )
        
        style.map(
            'Custom.TNotebook.Tab',
            background=[('selected', COLORS['tab_sel'])],
            foreground=[('selected', '#ffffff')]
        )
        
        self.notebook = ttk.Notebook(parent, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))
        
        # Create tabs
        self.image_preview_tab = tk.Frame(self.notebook, bg=COLORS['background'])
        self.model_tab = tk.Frame(self.notebook, bg=COLORS['background'])
        
        self.notebook.add(self.image_preview_tab, text='📊  Image Preview')
        self.notebook.add(self.model_tab, text='🤖  Load Model')
        
        # Build tab content
        self._build_image_preview_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
    def _build_image_preview_tab(self):
        """Build the Image Preview tab with NDSI calculation (left) and Model Prediction (right)"""
        # Create two-column layout
        columns_frame = tk.Frame(self.image_preview_tab, bg=COLORS['background'])
        columns_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # LEFT COLUMN - Image Processing (NDSI)
        left_column = tk.Frame(columns_frame, bg=COLORS['background'])
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # RIGHT COLUMN - Model Prediction
        right_column = tk.Frame(columns_frame, bg=COLORS['background'])
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Build left column (NDSI)
        self._build_ndsi_column(left_column)
        
        # Build right column (Model Prediction)
        self._build_prediction_column(right_column)
        
    def _build_ndsi_column(self, parent):
        """Build NDSI calculation column (LEFT)"""
        # Folders card
        folders_card_outer, folders_card = self._make_card(parent, "Folders", "📁")
        folders_card_outer.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            folders_card,
            text="Select input and output folders",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted']
        ).pack(fill='x', pady=(0, 10))
        
        # Input folder label
        tk.Label(
            folders_card,
            text="Input Folder:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 5))
        
        self.folder_label = tk.Label(
            folders_card,
            text="No folder selected",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w'
        )
        self.folder_label.pack(fill='x', pady=(0, 10))
        
        # Select input folder button
        tk.Button(
            folders_card,
            text="📂  Select Input Folder",
            font=_FONT_BTN,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=10,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.select_input_folder_ndsi
        ).pack(fill='x', pady=(0, 10))
        
        # Output folder label
        tk.Label(
            folders_card,
            text="Output Folder:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(10, 5))
        
        self.output_folder_label = tk.Label(
            folders_card,
            text="Not selected",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w'
        )
        self.output_folder_label.pack(fill='x', pady=(0, 10))
        
        # Select output folder button
        tk.Button(
            folders_card,
            text="💾  Select Output Folder",
            font=_FONT_BTN,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=10,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.select_output_folder_ndsi
        ).pack(fill='x')
        
        # Current Image card
        image_card_outer, image_card = self._make_card(parent, "Current Image", "📄")
        image_card_outer.pack(fill='x', pady=(0, 15))
        
        self.file_name_label = tk.Label(
            image_card,
            text="No image selected",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted']
        )
        self.file_name_label.pack(fill='x')
        
        # Calculation card
        calc_card_outer, calc_card = self._make_card(parent, "Calculation", "⚙️")
        calc_card_outer.pack(fill='x', pady=(0, 15))
        
        # Index type dropdown
        tk.Label(
            calc_card,
            text="Index Type:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 5))
        
        self.index_type = ttk.Combobox(
            calc_card,
            values=["NDSI", "NDVI", "NDWI"],
            state="readonly",
            font=_FONT_BODY
        )
        self.index_type.set("NDSI")
        self.index_type.pack(fill='x', pady=(0, 10))
        
        # NDSI Threshold
        tk.Label(
            calc_card,
            text="NDSI Threshold:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(10, 5))
        
        self.threshold_var = tk.DoubleVar(value=0.40)
        threshold_scale = tk.Scale(
            calc_card,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.threshold_var,
            bg=COLORS['card_bg'],
            fg=COLORS['text'],
            highlightthickness=0,
            troughcolor=COLORS['secondary_bg'],
            activebackground=COLORS['primary']
        )
        threshold_scale.pack(fill='x')
        
        # Run Calculation button
        tk.Button(
            calc_card,
            text="▶  Run Calculation",
            font=_FONT_BTN,
            bg=COLORS['primary'],
            fg='#ffffff',
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=12,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.run_calculation
        ).pack(fill='x', pady=(10, 0))
        
        # Actions card
        actions_card_outer, actions_card = self._make_card(parent, "Actions", "🎨")
        actions_card_outer.pack(fill='both', expand=True)
        
    def _build_prediction_column(self, parent):
        """Build Model Prediction column (RIGHT)"""
        # Machine Learning Model card
        ml_card_outer, ml_card = self._make_card(parent, "Machine Learning Model", "🤖")
        ml_card_outer.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            ml_card,
            text="Upload your trained Keras model (.keras, .h5) for classification",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            wraplength=300
        ).pack(fill='x', pady=(0, 10))
        
        # Selected Model label
        tk.Label(
            ml_card,
            text="Selected Model:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 5))
        
        self.model_path_label = tk.Label(
            ml_card,
            text="No model loaded",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w',
            wraplength=300
        )
        self.model_path_label.pack(fill='x', pady=(0, 10))
        
        # Button frame for model actions
        model_btn_frame = tk.Frame(ml_card, bg=COLORS['card_bg'])
        model_btn_frame.pack(fill='x')
        
        # Upload Model button
        tk.Button(
            model_btn_frame,
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
            command=self.upload_model
        ).pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Model Details button
        tk.Button(
            model_btn_frame,
            text="📊  Model Details",
            font=_FONT_SMALL,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=8,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.show_model_details
        ).pack(side='right', fill='x', expand=True, padx=(5, 0))
        
        # Sliding Window Settings card
        window_card_outer, window_card = self._make_card(parent, "Sliding Window Settings", "⚙️")
        window_card_outer.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            window_card,
            text="Window Size & Stride",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 10))
        
        # Window Size
        window_size_frame = tk.Frame(window_card, bg=COLORS['card_bg'])
        window_size_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            window_size_frame,
            text="Window Size:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(side='left')
        
        self.window_size_var = tk.IntVar(value=128)
        tk.Entry(
            window_size_frame,
            textvariable=self.window_size_var,
            font=_FONT_BODY,
            width=10,
            bg=COLORS['listbox_bg'],
            fg=COLORS['text'],
            insertbackground=COLORS['text']
        ).pack(side='left', padx=(10, 5))
        
        tk.Label(
            window_size_frame,
            text="px (square window)",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted']
        ).pack(side='left')
        
        # Stride
        stride_frame = tk.Frame(window_card, bg=COLORS['card_bg'])
        stride_frame.pack(fill='x')
        
        tk.Label(
            stride_frame,
            text="Stride:",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(side='left')
        
        self.stride_var = tk.IntVar(value=64)
        tk.Entry(
            stride_frame,
            textvariable=self.stride_var,
            font=_FONT_BODY,
            width=10,
            bg=COLORS['listbox_bg'],
            fg=COLORS['text'],
            insertbackground=COLORS['text']
        ).pack(side='left', padx=(10, 5))
        
        tk.Label(
            stride_frame,
            text="px (overlap = window_size - stride)",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted']
        ).pack(side='left')
        
        # Apply Settings button
        tk.Button(
            window_card,
            text="✓  Apply Settings",
            font=_FONT_SMALL,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=8,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.apply_window_settings
        ).pack(fill='x', pady=(10, 0))
        
        # Model Prediction folder selection card
        pred_folder_card_outer, pred_folder_card = self._make_card(parent, "Model Prediction", "📁")
        pred_folder_card_outer.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            pred_folder_card,
            text="Select input and output folders",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted']
        ).pack(fill='x', pady=(0, 10))
        
        # Select Input Folder button
        tk.Button(
            pred_folder_card,
            text="📂  Select Input Folder",
            font=_FONT_SMALL,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=8,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.select_input_folder_model
        ).pack(fill='x', pady=(0, 10))
        
        # Select Output Folder button
        tk.Button(
            pred_folder_card,
            text="💾  Select Output Folder",
            font=_FONT_SMALL,
            bg=COLORS['secondary_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['primary_hover'],
            activeforeground='#ffffff',
            pady=8,
            cursor="hand2",
            relief='flat',
            bd=0,
            command=self.select_output_folder_model
        ).pack(fill='x')
        
        # Folder display
        folders_display = tk.Frame(pred_folder_card, bg=COLORS['card_bg'])
        folders_display.pack(fill='x', pady=(10, 0))
        
        # Input Folder
        input_col = tk.Frame(folders_display, bg=COLORS['card_bg'])
        input_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(
            input_col,
            text="Input Folder:",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x')
        
        self.model_folder_label = tk.Label(
            input_col,
            text="No folder selected",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w'
        )
        self.model_folder_label.pack(fill='x')
        
        # Output Folder
        output_col = tk.Frame(folders_display, bg=COLORS['card_bg'])
        output_col.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        tk.Label(
            output_col,
            text="Output Folder:",
            font=("Segoe UI", 9, "bold"),
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x')
        
        self.model_output_folder_label = tk.Label(
            output_col,
            text="Not selected",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w'
        )
        self.model_output_folder_label.pack(fill='x')
        
        # Prediction Results card
        results_card_outer, results_card = self._make_card(parent, "Model Prediction", "🎯")
        results_card_outer.pack(fill='both', expand=True)
        
        tk.Label(
            results_card,
            text="Images in folder:",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 8))
        
        # Image listbox
        listbox_frame = tk.Frame(results_card, bg=COLORS['card_bg'])
        listbox_frame.pack(fill='x', pady=(0, 10))
        
        scrollbar = tk.Scrollbar(listbox_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        self.model_image_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=_FONT_BODY,
            bg=COLORS['listbox_bg'],
            fg=COLORS['text'],
            selectbackground=COLORS['listbox_sel'],
            height=5
        )
        self.model_image_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.model_image_listbox.yview)
        
        self.model_image_listbox.bind('<<ListboxSelect>>', self.load_selected_model_image)
        
        # Clear Selection button
        tk.Button(
            results_card,
            text="🔄  Clear Selection (Batch Mode)",
            font=_FONT_SMALL,
            bg=COLORS['card_bg'],
            fg=COLORS['text'],
            activebackground=COLORS['secondary_bg'],
            pady=6,
            cursor="hand2",
            relief='flat',
            command=self.clear_model_image_selection
        ).pack(fill='x', pady=(0, 15))
        
        # Current Image info
        tk.Label(
            results_card,
            text="📄  Current Image",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(fill='x', pady=(0, 8))
        
        self.model_file_name_label = tk.Label(
            results_card,
            text="No image selected",
            font=_FONT_BODY,
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            wraplength=280
        )
        self.model_file_name_label.pack(fill='x', pady=(0, 15))
        
        # Run Prediction button
        self.prediction_button = tk.Button(
            results_card,
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
            command=self.run_model_prediction
        )
        self.prediction_button.pack(fill='x')
        
    def _make_card(self, parent, title, icon=""):
        """Create a card-style container with title"""
        outer = tk.Frame(parent, bg=COLORS['card_border'], bd=0, highlightthickness=0)
        
        card = tk.Frame(outer, bg=COLORS['card_bg'], bd=0, highlightthickness=0)
        card.pack(fill='both', expand=True, padx=1, pady=1)
        
        header = tk.Frame(card, bg=COLORS['card_bg'])
        header.pack(fill='x', pady=(15, 10), padx=20)
        
        if icon:
            tk.Label(
                header,
                text=icon,
                font=("Segoe UI", 14),
                bg=COLORS['card_bg'],
                fg=COLORS['text']
            ).pack(side='left', padx=(0, 10))
        
        tk.Label(
            header,
            text=title,
            font=_FONT_HEADING,
            bg=COLORS['card_bg'],
            fg=COLORS['text']
        ).pack(side='left')
        
        content = tk.Frame(card, bg=COLORS['card_bg'])
        content.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        return outer, content
        
    def _create_status_bar(self, parent):
        """Create status bar"""
        status_frame = tk.Frame(parent, bg=COLORS['card_bg'], height=52)
        status_frame.pack(fill='x', pady=(14, 0))
        status_frame.pack_propagate(False)
        
        # Top border
        tk.Frame(status_frame, bg=COLORS['card_border'], height=1).pack(fill='x')
        
        self.status = tk.Label(
            status_frame,
            text="⚪  Ready – Select input folder to begin",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS['card_bg'],
            fg=COLORS['text_muted'],
            anchor='w',
            padx=20,
            pady=10
        )
        self.status.pack(fill='both', expand=True)
        
    def _create_footer(self, parent):
        """Create footer"""
        tk.Label(
            parent,
            text="Powered by Rasterio & Tkinter  |  Smart Band Detection  ©  2026",
            font=_FONT_SMALL,
            bg=COLORS['background'],
            fg=COLORS['text_muted']
        ).pack(pady=(10, 0))
        
    def _on_tab_changed(self, event):
        """Handle tab change event"""
        # Build model tab lazily if needed
        pass
        
    # ============================================================
    # FOLDER SELECTION METHODS
    # ============================================================
    
    def select_input_folder_ndsi(self):
        """Select input folder for NDSI calculation"""
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.folder_path = path
            self.folder_label.config(text=path)
            self.set_status("info", f"✓ Input folder selected: {os.path.basename(path)}")
            
    def select_output_folder_ndsi(self):
        """Select output folder for NDSI calculation"""
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder_path = path
            self.output_folder_label.config(text=path)
            self.set_status("info", f"✓ Output folder selected: {os.path.basename(path)}")
            
    def select_input_folder_model(self):
        """Select input folder for model prediction"""
        path = filedialog.askdirectory(title="Select Input Folder for Prediction")
        if path:
            self.model_folder_path = path
            self.model_folder_label.config(text=os.path.basename(path))
            
            # Load images in listbox
            images = [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff'))]
            self.model_image_listbox.delete(0, tk.END)
            for img in images:
                self.model_image_listbox.insert(tk.END, img)
                
            self.set_status("info", f"✓ Found {len(images)} images in folder")
            
    def select_output_folder_model(self):
        """Select output folder for model prediction"""
        path = filedialog.askdirectory(title="Select Output Folder for Predictions")
        if path:
            self.model_output_folder = path
            self.model_output_folder_label.config(text=os.path.basename(path))
            self.set_status("info", f"✓ Output folder selected: {os.path.basename(path)}")
            
    # ============================================================
    # MODEL METHODS
    # ============================================================
    
    def upload_model(self):
        """Upload and load a Keras model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras Models", "*.keras *.h5")]
        )
        
        if file_path:
            self.set_status("working", "Loading model...")
            success, message, model, input_size = self.model_predictor.load_model(file_path)
            
            if success:
                self.loaded_model = model
                self.model_input_size = input_size
                self.model_path_label.config(text=os.path.basename(file_path))
                
                # Update window size
                self.window_size_var.set(input_size)
                self.stride_var.set(input_size // 2)
                
                self.set_status("success", message)
            else:
                self.set_status("error", message)
                messagebox.showerror("Model Error", message)
                
    def show_model_details(self):
        """Show model details in a popup"""
        if not self.loaded_model:
            messagebox.showwarning("No Model", "Please upload a model first")
            return
            
        details = self.model_predictor.get_model_details(self.loaded_model)
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Model Details")
        popup.geometry("600x400")
        popup.configure(bg=COLORS['background'])
        
        text_widget = scrolledtext.ScrolledText(
            popup,
            font=("Courier", 10),
            bg=COLORS['card_bg'],
            fg=COLORS['text'],
            wrap=tk.WORD
        )
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert('1.0', details)
        text_widget.config(state='disabled')
        
    def apply_window_settings(self):
        """Apply sliding window settings"""
        window_size = self.window_size_var.get()
        stride = self.stride_var.get()
        
        self.set_status("info", f"✓ Window settings: {window_size}x{window_size}, Stride: {stride}")
        
    # ============================================================
    # IMAGE PROCESSING METHODS
    # ============================================================
    
    def run_calculation(self):
        """Run NDSI calculation"""
        if not self.folder_path:
            messagebox.showwarning("No Folder", "Please select input folder first")
            return
            
        if not self.output_folder_path:
            messagebox.showwarning("No Output", "Please select output folder first")
            return
            
        self.set_status("working", "Running NDSI calculation...")
        
        # Run calculation in background thread
        import threading
        thread = threading.Thread(
            target=self._run_calculation_thread,
            daemon=True
        )
        thread.start()
        
    def _run_calculation_thread(self):
        """Background thread for calculation"""
        index_type = self.index_type.get()
        threshold = self.threshold_var.get()
        
        success, message = self.image_processor.process_folder(
            self.folder_path,
            self.output_folder_path,
            index_type,
            threshold
        )
        
        if success:
            self.root.after(0, lambda: self.set_status("success", message))
        else:
            self.root.after(0, lambda: self.set_status("error", message))
            
    # ============================================================
    # MODEL PREDICTION METHODS
    # ============================================================
    
    def load_selected_model_image(self, event):
        """Load selected image from listbox"""
        selection = self.model_image_listbox.curselection()
        if selection:
            idx = selection[0]
            filename = self.model_image_listbox.get(idx)
            self.model_file_name_label.config(text=filename)
            self.model_batch_mode = False
        else:
            self.model_batch_mode = True
            
    def clear_model_image_selection(self):
        """Clear image selection for batch mode"""
        self.model_image_listbox.selection_clear(0, tk.END)
        self.model_file_name_label.config(text="Batch mode - All images")
        self.model_batch_mode = True
        
    def run_model_prediction(self):
        """Run model prediction"""
        if not self.loaded_model:
            messagebox.showwarning("No Model", "Please upload a model first")
            return
            
        if not self.model_folder_path:
            messagebox.showwarning("No Input", "Please select input folder")
            return
            
        if not self.model_output_folder:
            messagebox.showwarning("No Output", "Please select output folder")
            return
            
        self.set_status("working", "Running prediction...")
        
        # Run prediction in background thread
        thread = threading.Thread(
            target=self._run_prediction_thread,
            daemon=True
        )
        thread.start()
        
    def _run_prediction_thread(self):
        """Background thread for prediction"""
        window_size = self.window_size_var.get()
        stride = self.stride_var.get()
        
        if self.model_batch_mode:
            # Batch mode - process all images
            images = [f for f in os.listdir(self.model_folder_path) 
                     if f.lower().endswith(('.tif', '.tiff'))]
        else:
            # Single image mode
            selection = self.model_image_listbox.curselection()
            if selection:
                images = [self.model_image_listbox.get(selection[0])]
            else:
                images = []
                
        if not images:
            self.root.after(0, lambda: self.set_status("error", "No images to process"))
            return
            
        success, message = self.model_predictor.predict_batch(
            self.loaded_model,
            self.model_folder_path,
            self.model_output_folder,
            images,
            window_size,
            stride
        )
        
        if success:
            self.root.after(0, lambda: self.set_status("success", message))
        else:
            self.root.after(0, lambda: self.set_status("error", message))
            
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def set_status(self, status_type, message):
        """Update status bar with colored icon"""
        icons = {
            "info": "🔵",
            "success": "✅",
            "error": "❌",
            "working": "⚙️"
        }
        
        icon = icons.get(status_type, "⚪")
        self.status.config(text=f"{icon}  {message}")
        
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = RasterCalculatorGUI()
    app.run()
