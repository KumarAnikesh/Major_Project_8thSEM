"""
===============================================================================
IMAGE PROCESSING MODULE - NDSI Calculation & GeoTIFF Composite Creation
===============================================================================
Handles all image processing operations including:
- NDSI (Normalized Difference Snow Index) calculation
- NDVI (Normalized Difference Vegetation Index) calculation  
- NDWI (Normalized Difference Water Index) calculation
- Smart band detection for multi-spectral imagery
- GeoTIFF composite creation with geospatial metadata
===============================================================================
"""

import rasterio
import numpy as np
import os
from datetime import datetime


class ImageProcessor:
    """Handles all image processing operations for spectral indices"""
    
    def __init__(self):
        """Initialize the image processor"""
        pass
        
    def detect_bands_from_composite(self, img_path):
        """
        Automatically detect the correct Green and NIR (or SWIR) bands from composite image.
        
        Args:
            img_path: Path to the GeoTIFF file
            
        Returns:
            tuple: (green_band_idx, nir_or_swir_band_idx, total_bands, detection_msg)
        """
        try:
            with rasterio.open(img_path) as src:
                total_bands = src.count
                
                # Read all bands to check values
                bands_data = [src.read(i+1) for i in range(total_bands)]
                
                # Calculate mean reflectance for each band
                band_means = [np.nanmean(band) for band in bands_data]
                
                # Detection logic based on number of bands
                if total_bands == 4:
                    # Standard 4-band: R, G, B, NIR
                    green_idx = 2  # Green is band 2 (index 1)
                    nir_idx = 4    # NIR is band 4 (index 3)
                    msg = "✓ 4-band composite detected (R,G,B,NIR)"
                    
                elif total_bands == 3:
                    # Could be RGB or custom 3-band
                    # Assume band 2 is Green, band 3 is NIR
                    green_idx = 2
                    nir_idx = 3
                    msg = "✓ 3-band composite detected (assuming G,NIR configuration)"
                    
                elif total_bands >= 5:
                    # Multi-spectral with 5+ bands
                    # Use heuristic: Green typically has moderate reflectance,
                    # NIR/SWIR has higher reflectance
                    
                    # Find band with moderate mean (likely Green)
                    sorted_indices = np.argsort(band_means)
                    green_idx = sorted_indices[len(sorted_indices)//2] + 1  # Middle band
                    
                    # Find band with highest mean (likely NIR/SWIR)
                    nir_idx = sorted_indices[-1] + 1
                    
                    msg = f"✓ {total_bands}-band composite detected (auto-detected Green=Band{green_idx}, NIR/SWIR=Band{nir_idx})"
                    
                else:
                    # Fallback for 2-band or 1-band
                    green_idx = 1
                    nir_idx = min(2, total_bands)
                    msg = f"⚠️ {total_bands}-band composite (unusual, using Band{green_idx} & Band{nir_idx})"
                
                print(msg)
                print(f"   Band means: {[f'{m:.2f}' for m in band_means]}")
                
                return green_idx, nir_idx, total_bands, msg
                
        except Exception as e:
            print(f"❌ Error detecting bands: {e}")
            # Fallback defaults
            return 2, 4, 4, f"❌ Error detecting bands, using defaults"
    
    def calculate_ndsi(self, img_path, green_band=2, nir_band=4, auto_detect=True):
        """
        Calculate NDSI with smart band detection.
        
        NDSI = (Green - NIR) / (Green + NIR)
        
        Args:
            img_path: Path to GeoTIFF
            green_band: Green band index (default 2)
            nir_band: NIR/SWIR band index (default 4)
            auto_detect: Automatically detect bands (default True)
            
        Returns:
            tuple: (ndsi_array, metadata_dict)
        """
        try:
            with rasterio.open(img_path) as src:
                # Auto-detect bands if enabled
                if auto_detect:
                    green_band, nir_band, total_bands, msg = self.detect_bands_from_composite(img_path)
                
                # Read bands
                green = src.read(green_band).astype(float)
                nir = src.read(nir_band).astype(float)
                
                # Calculate NDSI
                # Handle division by zero
                denominator = green + nir
                denominator = np.where(denominator == 0, np.nan, denominator)
                
                ndsi = (green - nir) / denominator
                
                # Clip to valid range [-1, 1]
                ndsi = np.clip(ndsi, -1, 1)
                
                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': 1,
                    'dtype': 'float32',
                    'driver': 'GTiff',
                    'nodata': None
                }
                
                return ndsi, metadata
                
        except Exception as e:
            print(f"❌ Error calculating NDSI: {e}")
            raise
    
    def calculate_ndvi(self, img_path, red_band=1, nir_band=4):
        """
        Calculate NDVI (Normalized Difference Vegetation Index).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            img_path: Path to GeoTIFF
            red_band: Red band index
            nir_band: NIR band index
            
        Returns:
            tuple: (ndvi_array, metadata_dict)
        """
        try:
            with rasterio.open(img_path) as src:
                red = src.read(red_band).astype(float)
                nir = src.read(nir_band).astype(float)
                
                denominator = nir + red
                denominator = np.where(denominator == 0, np.nan, denominator)
                
                ndvi = (nir - red) / denominator
                ndvi = np.clip(ndvi, -1, 1)
                
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': 1,
                    'dtype': 'float32',
                    'driver': 'GTiff',
                    'nodata': None
                }
                
                return ndvi, metadata
                
        except Exception as e:
            print(f"❌ Error calculating NDVI: {e}")
            raise
    
    def calculate_ndwi(self, img_path, green_band=2, nir_band=4):
        """
        Calculate NDWI (Normalized Difference Water Index).
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            img_path: Path to GeoTIFF
            green_band: Green band index
            nir_band: NIR band index
            
        Returns:
            tuple: (ndwi_array, metadata_dict)
        """
        # NDWI uses same formula as NDSI
        return self.calculate_ndsi(img_path, green_band, nir_band)
    
    def save_geotiff(self, array, metadata, output_path):
        """
        Save array as GeoTIFF with metadata.
        
        Args:
            array: Numpy array to save
            metadata: Rasterio metadata dictionary
            output_path: Output file path
        """
        try:
            # Ensure array is 2D
            if len(array.shape) == 3:
                array = array[:, :, 0]
            
            # Handle NaN values
            array_clean = np.where(np.isfinite(array), array, -9999)
            
            # Update metadata for output
            metadata.update({
                'count': 1,
                'dtype': 'float32',
                'compress': 'lzw',
                'nodata': -9999
            })
            
            # Write file
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(array_clean.astype(np.float32), 1)
            
            print(f"✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving GeoTIFF: {e}")
            raise
    
    def create_composite_geotiff(self, img_path, output_path, index_type='NDSI', threshold=0.4):
        """
        Create composite GeoTIFF with spectral index calculation.
        
        Args:
            img_path: Input image path
            output_path: Output GeoTIFF path
            index_type: Type of index ('NDSI', 'NDVI', 'NDWI')
            threshold: Threshold value for binary mask
            
        Returns:
            tuple: (success, message)
        """
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(img_path)}")
            print(f"Index Type: {index_type}")
            print(f"Threshold: {threshold}")
            print(f"{'='*60}")
            
            # Calculate index
            if index_type == 'NDSI':
                index_array, metadata = self.calculate_ndsi(img_path, auto_detect=True)
            elif index_type == 'NDVI':
                index_array, metadata = self.calculate_ndvi(img_path)
            elif index_type == 'NDWI':
                index_array, metadata = self.calculate_ndwi(img_path)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Save result
            self.save_geotiff(index_array, metadata, output_path)
            
            # Calculate statistics
            valid_pixels = np.isfinite(index_array)
            mean_val = np.mean(index_array[valid_pixels])
            min_val = np.min(index_array[valid_pixels])
            max_val = np.max(index_array[valid_pixels])
            
            # Count pixels above threshold
            above_threshold = np.sum(index_array[valid_pixels] > threshold)
            total_valid = np.sum(valid_pixels)
            percentage = (above_threshold / total_valid * 100) if total_valid > 0 else 0
            
            print(f"\n📊 Statistics:")
            print(f"   Mean {index_type}: {mean_val:.4f}")
            print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"   Pixels > {threshold}: {above_threshold} ({percentage:.2f}%)")
            print(f"{'='*60}\n")
            
            return True, f"✓ {index_type} calculated successfully"
            
        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def process_folder(self, input_folder, output_folder, index_type='NDSI', threshold=0.4):
        """
        Process all GeoTIFF files in a folder.
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder path
            index_type: Type of index to calculate
            threshold: Threshold value
            
        Returns:
            tuple: (success, message)
        """
        try:
            # Get all TIFF files
            files = [f for f in os.listdir(input_folder) 
                    if f.lower().endswith(('.tif', '.tiff'))]
            
            if not files:
                return False, "No TIFF files found in input folder"
            
            print(f"\n🔍 Found {len(files)} TIFF files to process")
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each file
            success_count = 0
            for filename in files:
                input_path = os.path.join(input_folder, filename)
                
                # Create output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_{index_type.lower()}.tif"
                output_path = os.path.join(output_folder, output_filename)
                
                # Process file
                success, msg = self.create_composite_geotiff(
                    input_path,
                    output_path,
                    index_type,
                    threshold
                )
                
                if success:
                    success_count += 1
            
            # Summary
            if success_count == len(files):
                return True, f"✅ Successfully processed all {len(files)} files"
            else:
                return False, f"⚠️ Processed {success_count}/{len(files)} files (some errors)"
                
        except Exception as e:
            return False, f"❌ Error processing folder: {str(e)}"
    
    def read_composite_for_model(self, img_path):
        """
        Read composite GeoTIFF for model prediction.
        Returns raw data with minimal preprocessing.
        
        Args:
            img_path: Path to composite GeoTIFF
            
        Returns:
            tuple: (image_array, metadata)
        """
        try:
            with rasterio.open(img_path) as src:
                # Read all bands
                image = src.read()
                
                # Transpose to (H, W, C) format for model
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                
                # Convert to float32
                image = image.astype(np.float32)
                
                # Replace NaN/Inf with 0
                image = np.where(np.isfinite(image), image, 0.0)
                
                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height
                }
                
                return image, metadata
                
        except Exception as e:
            print(f"❌ Error reading composite: {e}")
            raise


# ============================================================
# MAIN ENTRY POINT (for testing)
# ============================================================
if __name__ == "__main__":
    # Test the image processor
    processor = ImageProcessor()
    print("Image Processing Module loaded successfully!")
    print("\nSupported operations:")
    print("  - NDSI (Normalized Difference Snow Index)")
    print("  - NDVI (Normalized Difference Vegetation Index)")
    print("  - NDWI (Normalized Difference Water Index)")
    print("  - Smart band detection")
    print("  - GeoTIFF composite creation")
