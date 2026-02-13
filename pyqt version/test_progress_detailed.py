#!/usr/bin/env python3
"""
MINIMAL PROGRESS BAR TEST - Simulates the exact same pattern as your app
"""

import sys
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QProgressBar, QLabel, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal

class SimulatedPredictionWorker(QThread):
    """Simulates the prediction worker with progress updates"""
    overall_progress = pyqtSignal(int)
    image_progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def run(self):
        try:
            # Simulate processing 255 windows (like your sliding window)
            total_windows = 255
            image_name = "test_image_composite.tif"
            
            self.log.emit(f"Starting processing {total_windows} windows...")
            self.overall_progress.emit(0)
            self.image_progress.emit(0, image_name)
            
            for i in range(total_windows + 1):
                pct = int((i * 100) / total_windows)
                
                # Emit progress every 1%
                if i == 0 or pct > int(((i-1) * 100) / total_windows):
                    self.overall_progress.emit(pct)
                    self.image_progress.emit(pct, image_name)
                    self.log.emit(f"Window {i}/{total_windows} - {pct}%")
                
                # Simulate work
                time.sleep(0.02)  # 20ms per window = ~5 seconds total
            
            self.overall_progress.emit(100)
            self.image_progress.emit(100, image_name)
            self.finished.emit(True, "Complete!")
            
        except Exception as e:
            self.finished.emit(False, str(e))

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Progress Bar Real-Time Test")
        self.setGeometry(100, 100, 600, 400)
        
        # Main widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Overall progress
        layout.addWidget(QLabel("🌍 Overall Progress:"))
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("%p%")
        layout.addWidget(self.overall_progress_bar)
        
        # Current image label
        self.current_image_label = QLabel("📄 Current Image: -")
        layout.addWidget(self.current_image_label)
        
        # Image progress
        self.image_progress_bar = QProgressBar()
        self.image_progress_bar.setRange(0, 100)
        self.image_progress_bar.setValue(0)
        self.image_progress_bar.setTextVisible(True)
        self.image_progress_bar.setFormat("%p%")
        layout.addWidget(self.image_progress_bar)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
        
        # Button
        self.btn = QPushButton("▶ Run Test")
        self.btn.clicked.connect(self.run_test)
        layout.addWidget(self.btn)
        
        self.worker = None
    
    def run_test(self):
        self.btn.setEnabled(False)
        self.overall_progress_bar.setValue(0)
        self.image_progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append("🚀 Starting test...")
        
        # Create worker
        self.worker = SimulatedPredictionWorker()
        self.worker.overall_progress.connect(self.on_overall_progress)
        self.worker.image_progress.connect(self.on_image_progress)
        self.worker.log.connect(self.log_text.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def on_overall_progress(self, percent):
        """Update overall progress bar"""
        print(f"Overall: {percent}%")  # Debug print
        self.overall_progress_bar.setValue(percent)
        self.overall_progress_bar.repaint()
    
    def on_image_progress(self, percent, image_name):
        """Update image progress bar"""
        print(f"Image: {percent}% - {image_name}")  # Debug print
        self.image_progress_bar.setValue(percent)
        self.current_image_label.setText(f"📄 Current Image: {image_name}")
        self.image_progress_bar.repaint()
        self.current_image_label.repaint()
    
    def on_finished(self, success, message):
        self.btn.setEnabled(True)
        self.log_text.append(f"\n{'✅' if success else '❌'} {message}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    print("="*60)
    print("PROGRESS BAR REAL-TIME UPDATE TEST")
    print("="*60)
    print("This test simulates 255 windows being processed")
    print("Watch the progress bars - they should update smoothly!")
    print("Also watch the terminal for debug output")
    print("="*60)
    
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
