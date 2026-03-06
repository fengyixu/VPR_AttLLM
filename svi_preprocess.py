import os
import sys
import shutil
import cv2
import json
import numpy as np
import glob
from PIL import Image
from pathlib import Path
from collections import defaultdict
from batch_controller import BatchController

import logging
logging.basicConfig(level=logging.INFO)

class ImgProcessor:
    def __init__(self):
        pass

    def resize_image(self, image_path, pixel_max=1024):
        """
        Resize image's longer side to pixel_max while maintaining aspect ratio and compress to reduce file size.
        
        Args:
            image_path (str): Path to the input image
            pixel_max (int): Maximum pixels for the longer side (default: 1024)
        
        Returns:
            PIL.Image: Resized and compressed image object
        """
        # Open the image
        img = Image.open(image_path)
        
        # Get current dimensions
        width, height = img.size
        
        # Calculate the scaling factor based on the longer side
        if width > height:
            scale_factor = pixel_max / width
        else:
            scale_factor = pixel_max / height
        
        # Only resize if the image is larger than pixel_max
        if scale_factor < 1:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert RGBA to RGB for JPEG compression (if needed)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        
        return img

    def resize_and_save(self, image_path, output_folder, pixel_max=1024):
        """
        Resize image and save to output_folder with optional suffix. Returns the saved image path.
        Args:
            image_path (str): Path to the input image
            output_folder (str): Folder to save the resized image
            pixel_max (int): Maximum pixels for the longer side
            suffix (str): Suffix to append to the filename before extension
        Returns:
            str: Path to the saved resized image
        """
        os.makedirs(output_folder, exist_ok=True)
        img = self.resize_image(image_path, pixel_max)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_folder, f"{base}.jpg")
        img.save(out_path, format="JPEG", quality=85, optimize=True)
        return out_path   

    def resize_image_pipeline(self, image_folder, output_folder, pixel_max=1024, max_workers=10):
        """
        Resize all images in a folder and save to output_folder using parallel processing.
        
        Args:
            image_folder: Path to folder containing input images
            output_folder: Path to folder for resized images
            pixel_max: Maximum pixels for the longer side (default: 1024)
            max_workers: Maximum number of parallel workers (default: 10)
        """
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all image files in the input folder
        image_files = []
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            if os.path.isfile(file_path) and Path(filename).suffix.lower() in image_extensions:
                image_files.append(Path(file_path))
        
        if not image_files:
            print(f"No image files found in {image_folder}")
            return
        
        def process_image(image_file):
            """Process a single image file."""
            item_key = image_file.name
            try:
                # Resize and save the image
                output_path = self.resize_and_save(str(image_file), output_folder, pixel_max)
                return {"status": "success", "file": item_key, "output": os.path.basename(output_path)}
            except Exception as e:
                return {"status": "error", "file": item_key, "error": str(e)}
        
        print(f"🚀 Starting parallel image resizing:")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total images to process: {len(image_files)}")
        print(f"   - Target pixel max: {pixel_max}")
        
        batch_controller = BatchController(
            save_dir=os.path.join(output_folder, "checkpoint"),
            checkpoint_interval=10
        )
        
        # Use parallel processing with rate limiting
        batch_controller.run_batch_parallel(
            items=image_files,
            process_func=process_image,
            common_params={},
            resume=True,
            desc="Resizing images in parallel",
            max_workers=max_workers
        )
        
        # Count results
        successful = sum(1 for result in batch_controller.results if result.get("status") == "success")
        failed = len(batch_controller.results) - successful
        
        print(f"\n✅ Image resizing complete!")
        print(f"   - Successfully processed: {successful} images")
        print(f"   - Failed to process: {failed} images")
        print(f"   - Output saved to: {output_folder}")

# ================================ 1. Image processing ==============================================

# # ------------------------------- 2. Mapillary panorama detection --------------------------------
#     def process_mapillary_panorama(self, mapillary_folder):
#         """
#         Detect if the image is a Mapillary panorama. Sample file name: 37.731942_-122.414787@pano@240910.jpg, 37.731332_-122.432448@162@160916.jpg
#         Collect all panorama paths and call _sample_perspective_views using batch parallel processing
#         """
#         # Create output folder if it doesn't exist
#         Path(mapillary_folder).mkdir(parents=True, exist_ok=True)
        
#         # Supported image extensions
#         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
#         # Find all panorama files in the folder
#         panorama_files = []
        
#         for filename in os.listdir(mapillary_folder):
#             file_path = os.path.join(mapillary_folder, filename)
#             if os.path.isfile(file_path) and Path(filename).suffix.lower() in image_extensions:
#                 # Check if it's a Mapillary panorama (contains @pano@ pattern)
#                 if '@pano@' in filename:
#                     panorama_files.append(Path(file_path))
        
#         if not panorama_files:
#             print(f"No Mapillary panorama files found in {mapillary_folder}")
#             return
        
#         def process_panorama(panorama_file):
#             """Process a single panorama file."""
#             item_key = panorama_file.name
#             try:
#                 # Call _sample_perspective_views for each panorama
#                 result = self._sample_perspective_views(
#                     str(panorama_file), 
#                     fov_deg=90, 
#                     sample_counts=1, 
#                     out_shape=(1024, 1024), 
#                     output_dir=mapillary_folder
#                 )
#                 if result:
#                     return {"status": "success", "file": item_key, "views_generated": len(result)}
#                 else:
#                     return {"status": "error", "file": item_key, "error": "Failed to generate perspective views"}
#             except Exception as e:
#                 return {"status": "error", "file": item_key, "error": str(e)}
        
#         print(f"🚀 Starting Mapillary panorama processing:")
#         print(f"   - Total panorama files found: {len(panorama_files)}")
        
#         batch_controller = BatchController(
#             save_dir=os.path.join(mapillary_folder, "checkpoint"),
#             checkpoint_interval=10
#         )
        
#         # Use parallel processing with rate limiting
#         batch_controller.run_batch_parallel(
#             items=panorama_files,
#             process_func=process_panorama,
#             common_params={},
#             resume=True,
#             desc="Processing Mapillary panoramas in parallel",
#             max_workers=10
#         )
        
#         # Count results
#         successful = sum(1 for result in batch_controller.results if result.get("status") == "success")
#         failed = len(panorama_files) - successful
#         total_views = sum(result.get("views_generated", 0) for result in batch_controller.results if result.get("status") == "success")
        
#         print(f"\n✅ Mapillary panorama processing complete!")
#         print(f"   - Successfully processed: {successful} panoramas")
#         print(f"   - Failed to process: {failed} panoramas")
#         print(f"   - Total perspective views generated: {total_views}")
#         print(f"   - Output saved to: {mapillary_folder}")

#     def _sample_perspective_views(self, panorama_path, fov_deg=90, sample_counts=1, out_shape=(1024, 1024), output_dir=None, tilt_deg=0, tilt_views=None):
#         """
#         Randomly sample panorama images to sample_counts perspective views with optional tilt angles. Call _helper_equirect2persp to generate the perspective views.
#         Args:
#             sample_counts: number of views to sample from each panorama
#             fov_deg: field of view in degrees
#         Save the sampled perspective views to original folder.The file name should replace @pano to @view_index. Example: 37.731942_-122.414787@pano@240910.jpg -> 37.731942_-122.414787@1@240910.jpg
#         After saving the sampled perspective views, delete the original panorama image.
#         """
#         import random
        
#         # Load panorama image
#         pano_img = cv2.imread(panorama_path, cv2.IMREAD_COLOR)
#         if pano_img is None:
#             print(f"Could not read panorama image: {panorama_path}")
#             return False
        
#         # Setup output directory (use original folder if not specified)
#         output_dir = output_dir if output_dir else os.path.dirname(panorama_path)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Parse the original filename to extract base name and extension
#         base_name = os.path.splitext(os.path.basename(panorama_path))[0]
#         extension = os.path.splitext(panorama_path)[1]
        
#         # Replace @pano with @view_index in the filename
#         if '@pano@' in base_name:
#             # Split by @pano@ to get the parts before and after
#             parts = base_name.split('@pano@')
#             if len(parts) == 2:
#                 prefix = parts[0]
#                 suffix = parts[1]
#             else:
#                 print(f"Invalid panorama filename format: {panorama_path}")
#                 return False
#         else:
#             print(f"Not a panorama file (missing @pano@): {panorama_path}")
#             return False
        
#         # Generate all possible yaw angles (360 degrees)
#         all_yaw_angles = np.linspace(0, 360, 36, endpoint=False)  # 36 views every 10 degrees
#         all_yaw_angles = all_yaw_angles[::-1]  # Reverse to make clockwise
        
#         # Randomly sample the specified number of views
#         if sample_counts > len(all_yaw_angles):
#             sample_counts = len(all_yaw_angles)
        
#         sampled_indices = random.sample(range(len(all_yaw_angles)), sample_counts)
#         output_files = []
        
#         for i, idx in enumerate(sampled_indices):
#             yaw = all_yaw_angles[idx]
            
#             # Determine tilt angle for this view
#             if tilt_views and i in tilt_views:
#                 pitch = tilt_views[i]
#             else:
#                 pitch = tilt_deg
            
#             # Generate perspective view with tilt
#             persp = self._helper_equirect2persp(pano_img, fov_deg, yaw, pitch, out_shape)
            
#             # Create filename with view index (1-based)
#             view_index = i + 1
#             new_filename = f"{prefix}@{view_index}@{suffix}{extension}"
#             out_path = os.path.join(output_dir, new_filename)
            
#             # Save the perspective view
#             cv2.imwrite(out_path, persp)
#             output_files.append(out_path)
        
#         # Delete the original panorama image
#         try:
#             os.remove(panorama_path)
#         except Exception as e:
#             print(f"Warning: Could not delete original panorama {panorama_path}: {e}")
        
#         return output_files

# ================================ 3. Panorama Conversion ==============================================
# ------------------------------- 3.1 Panorama 2 Perspective Views --------------------------------
    def convert_panorama_batch_pipeline(self, panorama_folder, output_folder, fov_deg, num_views, out_shape=(1024, 1024), tilt_deg=0, tilt_views=None, max_workers=10):
        """
        Convert all panorama images in the folder to perspective views.
        """
        from tqdm import tqdm
        import time
        
        os.makedirs(output_folder, exist_ok=True)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        panorama_path = Path(panorama_folder)
        
        # Simple tqdm progress while scanning entries
        print("🔍 Collecting panorama images...")
        image_files = [
            file_path for file_path in tqdm(
                list(panorama_path.iterdir()), desc="Scanning images", unit="file"
            )
            if file_path.is_file() and file_path.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {panorama_folder}")
            return
            
        print(f"✅ Found {len(image_files)} panorama images")
                
        def process_image(image_file):
            item_key = image_file.name
            try:
                # Add timing for individual image processing
                start_time = time.time()
                
                # Process the image with progress tracking
                self.extract_perspective_views(str(image_file), fov_deg, num_views, out_shape, output_dir=output_folder, tilt_deg=tilt_deg, tilt_views=tilt_views)
                processing_time = time.time() - start_time
                
                # Log slow processing for debugging
                if processing_time > 60:  # More than 1 minute
                    print(f"⚠️ Slow processing: {item_key} took {processing_time:.1f}s")
                
                return {"status": "success", "file": item_key, "processing_time": processing_time}
                    
            except Exception as e:
                print(f"✗ Error processing {item_key}: {str(e)}")
                return {"status": "error", "file": item_key, "error": str(e)}
            
        print(f"🚀 Starting parallel batch processing:")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total images to process: {len(image_files)}")
        print(f"   - Expected views per image: {num_views}")
        print(f"   - Output shape: {out_shape}")
            
        batch_controller = BatchController(
            save_dir=os.path.join(output_folder, "checkpoint"),
            checkpoint_interval=10
        )
        
        # Use parallel processing with rate limiting
        batch_controller.run_batch_parallel(
            items=image_files,
            process_func=process_image,
            common_params={},
            resume=True,
            desc="Converting panoramas to perspective views",
            max_workers=max_workers
        )

        # Calculate and display comprehensive results
        successful = sum(1 for result in batch_controller.results if result.get("status") == "success")
        failed = len(batch_controller.results) - successful
        total_views = successful * num_views
        avg_processing_time = sum(result.get("processing_time", 0) for result in batch_controller.results if result.get("status") == "success") / max(successful, 1)
        
        print(f"\n✅ Panorama conversion complete!")
        print(f"   - Successfully processed: {successful}/{len(image_files)} images")
        print(f"   - Failed: {failed} images")
        print(f"   - Total perspective views generated: {total_views}")
        print(f"   - Average processing time per image: {avg_processing_time:.2f}s")
        print(f"   - Output saved to: {output_folder}")

    def extract_perspective_views(self, panorama_path, fov_deg, num_views, out_shape=(1024, 1024), 
                                output_dir=None, tilt_deg=0, tilt_views=None):
        """
        Extract perspective views with parameters encoded in filenames.
        Filename format: basename@fov@yaw@pitch@height_width.ext
        
        Returns:
            list: File paths of perspective images with encoded parameters
        """
        from tqdm import tqdm
        
        # Load panorama image with progress indication
        pano_img = cv2.imread(panorama_path, cv2.IMREAD_COLOR)
        if pano_img is None:
            raise ValueError(f"Could not load image: {panorama_path}")
            
        base_name = os.path.splitext(os.path.basename(panorama_path))[0]
        out_dir = output_dir or os.path.join(os.path.dirname(panorama_path), f"{base_name}_extracted")
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate yaw angles (clockwise)
        yaw_angles = np.linspace(0, 360, num_views, endpoint=False)[::-1]
        
        file_paths = []
        
        # Use tqdm for view generation progress
        for i, yaw in enumerate(tqdm(yaw_angles, desc=f"Generating views for {base_name}", 
                                    unit="view", leave=False, disable=num_views <= 4)):
            # Determine pitch angle for this view
            pitch_deg = tilt_views.get(i, tilt_deg) if tilt_views else tilt_deg
            
            # Generate perspective view
            persp = self._helper_equirect2persp(pano_img, fov_deg, yaw, pitch_deg, out_shape)
            
            # Create filename with encoded parameters
            filename = f"{base_name}@{fov_deg}@{yaw:.1f}@{pitch_deg:+.1f}@{pano_img.shape[0]}x{pano_img.shape[1]}.jpg"
            filepath = os.path.join(out_dir, filename)
            
            # Save the perspective view with compression
            cv2.imwrite(filepath, persp, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            file_paths.append(filepath)
        
        return file_paths

    # [TODO] integrate the function to extract_perspective_views
    def _helper_equirect2persp(self, pano_img, fov_deg, yaw_deg, pitch_deg, out_shape):
        """Updated to use shared mapping logic"""
        uf, vf = self._get_perspective_mapping(fov_deg, yaw_deg, pitch_deg, pano_img.shape[:2], out_shape)
        return cv2.remap(pano_img, uf.astype(np.float32), vf.astype(np.float32), 
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

# ------------------------------- 3.1.1 Separate Perspective Views to subfolders --------------------------------
# Separate perspective views to subfolders based on the location and time.
# This is used for evaluation of the LLM-based VPR model.
    def separate_perspective_subfolders_pipeline(self, image_folder, yaw_dict={"folder_1": [0, 180], "folder_2": [90, 270]}):
        """
        Separate perspective views to subfolders based on yaw_deg. 
 
        Args:
            image_folder (str): Path to folder containing extracted panorama views
                sample filepath: filename = f"{base_name}@{fov_deg}@{yaw_deg:.1f}@{pitch_deg:+.1f}@{out_shape[0]}x{out_shape[1]}.jpg"
                sample base_name: 37.7660763_-122.442508738 or only 37.7660763_-122.442508738   
            yaw_dict (dict): Dictionary of subfolders and explicit yaw_deg values, e.g. {"folder_1": [0, 90, 180], "folder_2": [270]}
        """
        import os
        import shutil
        from collections import defaultdict
        
        # Create subfolders
        subfolders = {}
        for folder_name in yaw_dict.keys():
            subfolder_path = os.path.join(image_folder, folder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            subfolders[folder_name] = subfolder_path
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        if not image_files:
            print(f"⚠️ No image files found in {image_folder}")
            return
        
        # Define the processing function for batch processing
        def process_image_file(filename):
            try:
                # Parse filename to get yaw_deg
                filepath = os.path.join(image_folder, filename)
                parsed = self.parse_perspective_filename(filepath)
                yaw_deg = parsed['yaw_deg']
                
                # Determine which subfolder this file belongs to based on exact degree match
                target_subfolder = None
                for folder_name, yaw_degrees in yaw_dict.items():
                    if yaw_deg in yaw_degrees:
                        target_subfolder = folder_name
                        break
                
                if target_subfolder:
                    # Move file to appropriate subfolder
                    src_path = os.path.join(image_folder, filename)
                    dst_path = os.path.join(subfolders[target_subfolder], filename)
                    shutil.move(src_path, dst_path)
                    
                    return {
                        "status": "success",
                        "filename": filename,
                        "yaw_deg": yaw_deg,
                        "target_subfolder": target_subfolder
                    }
                else:
                    return {
                        "status": "skipped",
                        "filename": filename,
                        "yaw_deg": yaw_deg,
                        "reason": "yaw_deg not in any defined degree list"
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "filename": filename,
                    "error": str(e)
                }
        
        # Use batch processing for efficient file movement
        batch_controller = BatchController(
            save_dir=os.path.join(image_folder, "batch_logs"),
            checkpoint_interval=50
        )
        
        results = batch_controller.run_batch_parallel(
            items=image_files,
            process_func=process_image_file,
            desc="Separating perspective views",
            max_workers=20
        )
        
        # Print summary
        successful = sum(1 for result in results if result.get("status") == "success")
        skipped = sum(1 for result in results if result.get("status") == "skipped")
        errors = sum(1 for result in results if result.get("status") == "error")
        
        print(f"✅ Perspective separation complete:")
        print(f"   - Total files processed: {len(results)}")
        print(f"   - Successfully moved: {successful}")
        print(f"   - Skipped (no matching degree): {skipped}")
        print(f"   - Errors: {errors}")
        
        # Print subfolder statistics
        subfolder_counts = defaultdict(int)
        for result in results:
            if result.get("status") == "success":
                subfolder_counts[result.get("target_subfolder")] += 1
        
        print(f"   - Files per subfolder:")
        for folder_name, count in subfolder_counts.items():
            print(f"     {folder_name}: {count} files")
        
        return results

    def merge_perspective_subfolders(self, image_folder):
        """
        Extract all files from subfolders to the parent folder and delete the original subfolders.
        
        Args:
            image_folder (str): Path to image folder that contains subfolders to be merged
        """
        import os
        import shutil
        
        if not os.path.exists(image_folder):
            print(f"⚠️ Folder not found: {image_folder}")
            return
        
        print(f"🔄 Processing folder: {image_folder}")
        
        # Get all subdirectories (excluding hidden files and batch log directories)
        subdirs = []
        for item in os.listdir(image_folder):
            item_path = os.path.join(image_folder, item)
            if (os.path.isdir(item_path) and 
                not item.startswith('.') and 
                not item.endswith('_logs')):
                subdirs.append(item)
        
        if not subdirs:
            print(f"   No subdirectories found in {image_folder}")
            return
        
        # Collect all image files from subdirectories
        files_to_move = []
        for subdir in subdirs:
            subdir_path = os.path.join(image_folder, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jpg'):
                    src_path = os.path.join(subdir_path, filename)
                    dst_path = os.path.join(image_folder, filename)
                    files_to_move.append((src_path, dst_path, subdir))
        
        if not files_to_move:
            print(f"   No image files found in subdirectories of {image_folder}")
            return
        
        # Define the processing function for batch processing
        def move_file(file_info):
            src_path, dst_path, subdir = file_info
            try:
                # Check if destination file already exists
                if os.path.exists(dst_path):
                    # Generate unique filename
                    base_name = os.path.splitext(os.path.basename(dst_path))[0]
                    ext = os.path.splitext(dst_path)[1]
                    counter = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(os.path.dirname(dst_path), f"{base_name}_{counter}{ext}")
                        counter += 1
                
                # Move the file
                shutil.move(src_path, dst_path)
                
                return {
                    "status": "success",
                    "filename": os.path.basename(src_path),
                    "from_subdir": subdir
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "filename": os.path.basename(src_path),
                    "from_subdir": subdir,
                    "error": str(e)
                }
        
        # Use batch processing for efficient file movement
        batch_controller = BatchController(
            save_dir=os.path.join(image_folder, "merge_batch_logs"),
            checkpoint_interval=50
        )
        
        results = batch_controller.run_batch_parallel(
            items=files_to_move,
            process_func=move_file,
            desc=f"Merging files from {os.path.basename(image_folder)}",
            max_workers=10
        )
        
        # Delete empty subdirectories
        deleted_subdirs = []
        for subdir in subdirs:
            subdir_path = os.path.join(image_folder, subdir)
            try:
                if os.path.exists(subdir_path) and not os.listdir(subdir_path):
                    os.rmdir(subdir_path)
                    deleted_subdirs.append(subdir)
            except Exception as e:
                print(f"   Warning: Could not delete subdirectory {subdir}: {e}")
        
        # Print summary
        successful = sum(1 for result in results if result.get("status") == "success")
        errors = sum(1 for result in results if result.get("status") == "error")
        
        print(f"✅ Merge complete:")
        print(f"   - Files moved: {successful}")
        print(f"   - Errors: {errors}")
        print(f"   - Subdirectories deleted: {len(deleted_subdirs)}")
        
        return results

# ------------------------------- 3.1.2 Tilt view sampling --------------------------------
# Sample original and tilted views for each location and organize them into separate folders.
    def sample_tilt(self, image_folder, original_views=[2, 4, 6, 8], tilt_views=[1, 3, 5, 7], sample_num=2, num_views=8):
        """
        Sample original and tilted views for each location and organize them into separate folders.
        
        Args:
            image_folder (str): Path to folder containing extracted panorama views
            original_views (list): List of view indices for original (non-tilted) views
            tilt_views (list): List of view indices for tilted views
            sample_num (int): Number of views to sample from each category
            num_views (int): Total number of views per panorama (default: 8)
        
        Returns:
            dict: Summary of sampling results
        """
        import random
        from collections import defaultdict
        
        # Create output directories
        original_dir = os.path.join(image_folder, f"{os.path.basename(image_folder)}_originalview")
        tilted_dir = os.path.join(image_folder, f"{os.path.basename(image_folder)}_newview")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(tilted_dir, exist_ok=True)
        
        # Group files by location (coordinates)
        location_files = defaultdict(lambda: {'original': [], 'tilted': []})
        
        # Scan all files in the folder
        for filename in os.listdir(image_folder):
            if not filename.endswith('.jpg'):
                continue
                
            # Parse filename: 37.7660763_-122.442508738_1_tilt-15.jpg
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) < 3:
                continue
                
            # Extract coordinates and view index
            coords = f"{parts[0]}_{parts[1]}"
            try:
                view_index = int(parts[2])
            except ValueError:
                continue
            
            # Categorize based on view index lists
            if view_index in original_views:
                location_files[coords]['original'].append(filename)
            elif view_index in tilt_views:
                location_files[coords]['tilted'].append(filename)
        
        # Process each location
        results = {'locations_processed': 0, 'original_sampled': 0, 'tilted_sampled': 0}
        
        for coords, files in location_files.items():
            # Sample original views
            available_original = files['original']
            if len(available_original) >= sample_num:
                sampled_original = random.sample(available_original, sample_num)
                for filename in sampled_original:
                    src_path = os.path.join(image_folder, filename)
                    dst_path = os.path.join(original_dir, filename)
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    results['original_sampled'] += 1
            
            # Sample tilted views
            available_tilted = files['tilted']
            if len(available_tilted) >= sample_num:
                sampled_tilted = random.sample(available_tilted, sample_num)
                for filename in sampled_tilted:
                    src_path = os.path.join(image_folder, filename)
                    dst_path = os.path.join(tilted_dir, filename)
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    results['tilted_sampled'] += 1
            
            results['locations_processed'] += 1
        
        print(f"✅ Sampling complete:")
        print(f"   - Locations processed: {results['locations_processed']}")
        print(f"   - Original views sampled: {results['original_sampled']}")
        print(f"   - Tilted views sampled: {results['tilted_sampled']}")
        print(f"   - Original views saved to: {original_dir}")
        print(f"   - Tilted views saved to: {tilted_dir}")
        
        return results

# ------------------------------- 3.2 Perspective 2 Panorama --------------------------------
# Reconstruct panorama from perspective views using known generation parameters.

    def batch_reconstruct_panoramas(self, views_dir, output_dir=None, max_pano_size=16384, memory_limit_gb=4, max_workers=10):
        """
        Reconstruct all panoramas from grouped perspective views using parallel processing.
        
        Args:
            views_dir: Directory containing perspective view files
            output_dir: Output directory for reconstructed panoramas
            max_pano_size: Maximum panorama dimension to prevent memory issues
            memory_limit_gb: Memory limit in GB for processing
            max_workers: Maximum number of parallel workers
        
        Returns:
            dict: {base_name: reconstructed_panorama_array}
        """
        import psutil
        
        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < memory_limit_gb:
            print(f"Warning: Available memory ({available_memory_gb:.1f}GB) is below limit ({memory_limit_gb}GB)")
        
        # Collect and group view files by base name
        view_files = glob.glob(os.path.join(views_dir, "*@*@*@*@*.jpg"))
        
        if not view_files:
            raise ValueError(f"No perspective view files found in {views_dir}")
        
        # Group files by base name
        view_groups = defaultdict(list)
        for filepath in view_files:
            try:
                params = self.parse_perspective_filename(filepath)
                view_groups[params['base_name']].append(filepath)
            except Exception as e:
                print(f"Warning: Skipping invalid filename {filepath}: {e}")
        
        if not view_groups:
            print(f"No valid panorama groups found in {views_dir}")
            return {}
        
        print(f"Found {len(view_groups)} panorama groups with {len(view_files)} total views")
        
        # Setup output directory
        output_dir = output_dir or os.path.join(views_dir, "reconstructed")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare items for parallel processing - use base_names as hashable items
        panorama_items = list(view_groups.keys())  # List of base_names (strings are hashable)
        
        # Create a mapping for easy lookup
        panorama_data = {}
        for base_name, group_files in view_groups.items():
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            panorama_data[base_name] = {
                'view_files': group_files,
                'output_path': output_path
            }
        
        def process_panorama(base_name):
            """Process a single panorama reconstruction."""
            data = panorama_data[base_name]
            view_files = data['view_files']
            output_path = data['output_path']
            
            try:
                panorama = self.reconstruct_panorama_from_views_optimized(
                    view_files, output_path, max_pano_size, memory_limit_gb
                )
                return {
                    "status": "success", 
                    "base_name": base_name, 
                    "views_processed": len(view_files),
                    "panorama": panorama
                }
            except Exception as e:
                return {
                    "status": "error", 
                    "base_name": base_name, 
                    "error": str(e)
                }
        
        print(f"🚀 Starting parallel panorama reconstruction:")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total panorama groups to process: {len(panorama_items)}")
        print(f"   - Memory limit: {memory_limit_gb}GB, Max panorama size: {max_pano_size}")
        
        batch_controller = BatchController(
            save_dir=os.path.join(output_dir, "checkpoint"),
            checkpoint_interval=5
        )
        
        # Use parallel processing with rate limiting
        batch_controller.run_batch_parallel(
            items=panorama_items,
            process_func=process_panorama,
            common_params={},
            resume=True,
            desc="Reconstructing panoramas in parallel",
            max_workers=max_workers
        )
        
        # Collect results
        results = {}
        successful = 0
        failed = 0
        
        for result in batch_controller.results:
            if result.get("status") == "success":
                results[result["base_name"]] = result["panorama"]
                successful += 1
            else:
                print(f"✗ Failed to reconstruct '{result.get('base_name', 'unknown')}': {result.get('error', 'Unknown error')}")
                failed += 1
        
        print(f"\n✅ Panorama reconstruction complete!")
        print(f"   - Successfully reconstructed: {successful} panoramas")
        print(f"   - Failed to reconstruct: {failed} panoramas")
        print(f"   - Output saved to: {output_dir}")
        
        return results

    def reconstruct_panorama_from_views_optimized(self, view_files, output_path=None, max_pano_size=16384, memory_limit_gb=4):
        """
        Memory-efficient panorama reconstruction with automatic downsampling.
        
        Args:
            view_files: List of perspective view file paths (same base_name group)
            output_path: Save reconstructed panorama (None = return only)
            max_pano_size: Maximum panorama dimension
            memory_limit_gb: Memory limit in GB
        
        Returns:
            numpy.ndarray: Reconstructed panorama image
        """
        import psutil
        
        if not view_files:
            raise ValueError("No view files provided")
        
        # Extract parameters from first file to determine panorama shape
        first_params = self.parse_perspective_filename(view_files[0])
        original_shape = first_params['shape']
        
        # Check if we need to downsample for memory efficiency
        pano_h, pano_w = original_shape
        estimated_memory_gb = (pano_h * pano_w * 3 * 4) / (1024**3)  # float32 for 3 channels
        
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Determine if downsampling is needed
        scale_factor = 1.0
        if estimated_memory_gb > memory_limit_gb or max(pano_h, pano_w) > max_pano_size:
            # Calculate appropriate scale factor
            max_dim = max(pano_h, pano_w)
            if max_dim > max_pano_size:
                scale_factor = max_pano_size / max_dim
            else:
                scale_factor = min(1.0, memory_limit_gb / estimated_memory_gb)
            
            # Ensure scale_factor is reasonable (not too small)
            scale_factor = max(scale_factor, 0.25)
            
            # Calculate new dimensions
            new_h = int(pano_h * scale_factor)
            new_w = int(pano_w * scale_factor)
            
            logging.debug(f"Downsampling panorama from {pano_w}x{pano_h} to {new_w}x{new_h} "
                  f"(scale: {scale_factor:.2f}, estimated memory: {estimated_memory_gb:.1f}GB)")
            
            pano_shape = (new_h, new_w)
        else:
            pano_shape = original_shape
            logging.debug(f"Processing panorama at {pano_w}x{pano_h} (estimated memory: {estimated_memory_gb:.1f}GB)")
        
        # Initialize accumulation arrays with smaller data type for memory efficiency
        pano_h, pano_w = pano_shape
        panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_map = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        # Process each view file
        processed_views = 0
        for i, filepath in enumerate(view_files):
            try:
                params = self.parse_perspective_filename(filepath)
                persp_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                
                if persp_img is None:
                    logging.warning(f"Could not load image {filepath}")
                    continue
                
                # Downsample perspective image if needed
                if scale_factor < 1.0:
                    persp_h, persp_w = persp_img.shape[:2]
                    new_persp_h = int(persp_h * scale_factor)
                    new_persp_w = int(persp_w * scale_factor)
                    persp_img = cv2.resize(persp_img, (new_persp_w, new_persp_h), interpolation=cv2.INTER_LINEAR)
                
                persp_img = persp_img.astype(np.float32)
                
                # Project back to panorama coordinates
                contribution, weights = self._helper_persp2equirect_optimized(
                    persp_img, params['fov_deg'], params['yaw_deg'], 
                    params['pitch_deg'], pano_shape
                )
                
                # Accumulate weighted contributions
                panorama += contribution * weights[..., np.newaxis]
                weight_map += weights
                processed_views += 1
                
                # Memory cleanup
                del contribution, weights
                
            except Exception as e:
                print(f"Warning: Failed to process {filepath}: {e}")
        
        if processed_views == 0:
            raise ValueError("No views were successfully processed")
        
        # Normalize by accumulated weights
        valid_mask = weight_map > 0
        if not np.any(valid_mask):
            raise ValueError("No valid pixels found in reconstruction")
        
        panorama[valid_mask] = panorama[valid_mask] / weight_map[valid_mask, np.newaxis]
        result = panorama.astype(np.uint8)
        
        # Save and report results
        if output_path:
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            coverage = np.mean(valid_mask) * 100
            logging.debug(f"Saved: {output_path} (coverage: {coverage:.1f}%, {processed_views}/{len(view_files)} views)")
        
        return result

    def _helper_persp2equirect_optimized(self, persp_img, fov_deg, yaw_deg, pitch_deg, pano_shape):
        """
        Memory-optimized projection of perspective image back to equirectangular coordinates.
        Uses chunked processing for large panoramas.
        """
        pano_h, pano_w = pano_shape
        persp_h, persp_w = persp_img.shape[:2]
        
        # Convert angles to radians
        fov, yaw, pitch = np.deg2rad([fov_deg, yaw_deg, pitch_deg])
        
        # Camera intrinsics
        fx = fy = persp_w / (2 * np.tan(fov / 2))
        cx, cy = persp_w / 2, persp_h / 2
        
        # Use chunked processing for large panoramas
        chunk_size = 512  # Process in 512x512 chunks
        
        # Initialize output arrays
        contribution = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weights = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        # Process in chunks
        for y_start in range(0, pano_h, chunk_size):
            y_end = min(y_start + chunk_size, pano_h)
            for x_start in range(0, pano_w, chunk_size):
                x_end = min(x_start + chunk_size, pano_w)
                
                # Create coordinate grids for this chunk
                lon = np.linspace(-np.pi + 2*np.pi*x_start/pano_w, 
                                -np.pi + 2*np.pi*x_end/pano_w, x_end-x_start, endpoint=False)
                lat = np.linspace(np.pi/2 - np.pi*y_start/pano_h, 
                                np.pi/2 - np.pi*y_end/pano_h, y_end-y_start)
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                
                # Convert to 3D unit vectors
                x = np.cos(lat_grid) * np.sin(lon_grid)
                y = np.sin(lat_grid)  
                z = np.cos(lat_grid) * np.cos(lon_grid)
                world_rays = np.stack([x, y, z], axis=2)
                
                # Inverse camera rotation (world to camera)
                Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                            [0, 1, 0],
                            [np.sin(yaw), 0, np.cos(yaw)]])
                Rx = np.array([[1, 0, 0],
                            [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)]])
                R_inv = (Ry @ Rx).T
                
                # Transform rays to camera coordinates
                cam_rays = world_rays @ R_inv.T
                
                # Project to image plane (only points in front of camera)
                valid_depth = cam_rays[..., 2] > 0
                
                u = cam_rays[..., 0] / cam_rays[..., 2] * fx + cx
                v = -cam_rays[..., 1] / cam_rays[..., 2] * fy + cy
                
                # Check bounds
                in_bounds = (valid_depth & (u >= 0) & (u < persp_w) & (v >= 0) & (v < persp_h))
                
                # Sample from perspective image for this chunk
                if np.any(in_bounds):
                    chunk_contribution = np.zeros((y_end-y_start, x_end-x_start, 3), dtype=np.float32)
                    chunk_weights = np.zeros((y_end-y_start, x_end-x_start), dtype=np.float32)
                    
                    chunk_contribution[in_bounds] = cv2.remap(
                        persp_img, u.astype(np.float32), v.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                    )[in_bounds]
                    chunk_weights[in_bounds] = 1.0
                    
                    # Add to main arrays
                    contribution[y_start:y_end, x_start:x_end] = chunk_contribution
                    weights[y_start:y_end, x_start:x_end] = chunk_weights
        
        return contribution, weights

    def reconstruct_panorama_from_views(self, view_files, output_path=None):
        """
        Reconstruct panorama from a list of perspective view files.
        All parameters are automatically extracted from filenames.
        
        Args:
            view_files: List of perspective view file paths (same base_name group)
            output_path: Save reconstructed panorama (None = return only)
        
        Returns:
            numpy.ndarray: Reconstructed panorama image
        """
        if not view_files:
            raise ValueError("No view files provided")
        
        # Extract parameters from first file to determine panorama shape
        first_params = self.parse_perspective_filename(view_files[0])
        pano_shape = first_params['shape']
        
        logging.debug(f"Reconstructing to {pano_shape[1]}x{pano_shape[0]} panorama")
        
        # Initialize accumulation arrays
        pano_h, pano_w = pano_shape
        panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_map = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        # Process each view file
        processed_views = 0
        for filepath in view_files:
            try:
                params = self.parse_perspective_filename(filepath)
                persp_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                
                if persp_img is None:
                    print(f"Warning: Could not load image {filepath}")
                    continue
                    
                persp_img = persp_img.astype(np.float32)
                
                # Project back to panorama coordinates
                contribution, weights = self._helper_persp2equirect(
                    persp_img, params['fov_deg'], params['yaw_deg'], 
                    params['pitch_deg'], params['shape']
                )
                
                # Accumulate weighted contributions
                panorama += contribution * weights[..., np.newaxis]
                weight_map += weights
                processed_views += 1
                
            except Exception as e:
                print(f"Warning: Failed to process {filepath}: {e}")
        
        if processed_views == 0:
            raise ValueError("No views were successfully processed")
        
        # Normalize by accumulated weights
        valid_mask = weight_map > 0
        if not np.any(valid_mask):
            raise ValueError("No valid pixels found in reconstruction")
        
        panorama[valid_mask] = panorama[valid_mask] / weight_map[valid_mask, np.newaxis]
        result = panorama.astype(np.uint8)
        
        # Save and report results
        if output_path:
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            coverage = np.mean(valid_mask) * 100
            print(f"Saved: {output_path} (coverage: {coverage:.1f}%, {processed_views}/{len(view_files)} views)")
        
        return result

    def _helper_persp2equirect(self, persp_img, fov_deg, yaw_deg, pitch_deg, pano_shape):
        """Project perspective image back to equirectangular coordinates."""
        pano_h, pano_w = pano_shape
        persp_h, persp_w = persp_img.shape[:2]
        
        # Convert angles to radians
        fov, yaw, pitch = np.deg2rad([fov_deg, yaw_deg, pitch_deg])
        
        # Camera intrinsics
        fx = fy = persp_w / (2 * np.tan(fov / 2))
        cx, cy = persp_w / 2, persp_h / 2
        
        # Panorama coordinate grids
        lon = np.linspace(-np.pi, np.pi, pano_w, endpoint=False)
        lat = np.linspace(np.pi/2, -np.pi/2, pano_h)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Convert to 3D unit vectors
        x = np.cos(lat_grid) * np.sin(lon_grid)
        y = np.sin(lat_grid)  
        z = np.cos(lat_grid) * np.cos(lon_grid)
        world_rays = np.stack([x, y, z], axis=2)
        
        # Inverse camera rotation (world to camera)
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                    [0, 1, 0],
                    [np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
        R_inv = (Ry @ Rx).T
        
        # Transform rays to camera coordinates
        cam_rays = world_rays @ R_inv.T
        
        # Project to image plane (only points in front of camera)
        valid_depth = cam_rays[..., 2] > 0
        
        u = cam_rays[..., 0] / cam_rays[..., 2] * fx + cx
        v = -cam_rays[..., 1] / cam_rays[..., 2] * fy + cy
        
        # Check bounds
        in_bounds = (valid_depth & (u >= 0) & (u < persp_w) & (v >= 0) & (v < persp_h))
        
        # Sample from perspective image
        contribution = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weights = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        if np.any(in_bounds):
            contribution[in_bounds] = cv2.remap(
                persp_img, u.astype(np.float32), v.astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )[in_bounds]
            weights[in_bounds] = 1.0
        
        return contribution, weights

    # ===== BBOX PANO RECONSTRUCTION AND TRACKING FUNCTIONS =====

    def reconstruct_panorama_bboxes(self, view_filename, bboxes_dict, persp_shape=(1024, 1024)):
        """
        Reconstruct bounding boxes in panorama coordinates from perspective view annotations.
        
        Args:
            view_filename: view filename, e.g. 37.7660763_-122.442508738@2014-08@90.0@+0.0@1024x1024.jpg
            bboxes_dict: Dict with bbox data arrays
                        Format: {'node_id': [...], 'annotation_bbox': [...], 'phrases': [...], 'confidence': [...]}
            persp_shape: (H, W) tuple of perspective view shape
            
        Returns:
            list: panorama_bboxes with transformed coordinates
        """
        # Validate input bboxes_dict
        required_keys = ['node_id', 'annotation_bbox', 'phrases', 'confidence']
        if not all(key in bboxes_dict for key in required_keys):
            logging.warning(f"Missing required keys in bboxes_dict for {view_filename}")
            return []
        
        # Check if arrays have the same length
        array_lengths = [len(bboxes_dict[key]) for key in required_keys]
        if len(set(array_lengths)) > 1:
            logging.warning(f"Inconsistent array lengths in bboxes_dict for {view_filename}: {dict(zip(required_keys, array_lengths))}")
            return []
        
        # Parse view parameters from filename
        try:
            params = self.parse_perspective_filename(view_filename)
            pano_shape = params['shape']
        except Exception as e:
            logging.warning(f"Failed to parse filename {view_filename}: {e}")
            return []
        
        panorama_bboxes = []
        
        # Zip all arrays together for elegant iteration
        bbox_data = zip(
            bboxes_dict['node_id'],
            bboxes_dict['annotation_bbox'], 
            bboxes_dict['phrases'],
            bboxes_dict['confidence']
        )
        
        # Transform each bbox to panorama coordinates
        for node_id, bbox, phrase, conf in bbox_data:
            try:
                pano_bbox_coords = self._transform_bbox_persp2pano(
                    bbox, params['fov_deg'], params['yaw_deg'], 
                    params['pitch_deg'], persp_shape, pano_shape
                )
                
                if pano_bbox_coords is not None:
                    panorama_bboxes.append({
                        'node_id': node_id,
                        'phrase': phrase,
                        'coords': pano_bbox_coords,
                        'source_view': view_filename,
                        'original_bbox': bbox,
                        'confidence': conf
                    })
            except Exception as e:
                logging.warning(f"Failed to transform bbox for {view_filename}, node_id {node_id}: {e}")
                continue
        
        return panorama_bboxes

    def _transform_bbox_persp2pano(self, bbox, fov_deg, yaw_deg, pitch_deg, persp_shape, pano_shape):
        """Transform bounding box from perspective view to panorama coordinates."""
        x1, y1, x2, y2 = bbox
        persp_h, persp_w = persp_shape
        
        # Sample points along bbox boundary for accurate transformation
        boundary_points = []
        
        # Sample more densely for better accuracy
        sample_density = max(5, min(20, int(max(x2-x1, y2-y1) / 10)))
        
        # Top edge
        for x in np.linspace(x1, x2, sample_density):
            boundary_points.append([x, y1])
        
        # Right edge  
        for y in np.linspace(y1, y2, sample_density):
            boundary_points.append([x2, y])
            
        # Bottom edge
        for x in np.linspace(x2, x1, sample_density):
            boundary_points.append([x, y2])
            
        # Left edge
        for y in np.linspace(y2, y1, sample_density):
            boundary_points.append([x1, y])
        
        # Transform each boundary point to panorama coordinates
        pano_coords = []
        for point in boundary_points:
            pano_coord = self._transform_point_persp2pano_compatible(
                point, fov_deg, yaw_deg, pitch_deg, persp_shape, pano_shape
            )
            if pano_coord is not None:
                pano_coords.append(pano_coord)
        
        return pano_coords if pano_coords else None
    
    def _transform_point_persp2pano_compatible(self, point, fov_deg, yaw_deg, pitch_deg, persp_shape, pano_shape):
        """Transform point from perspective to panorama using your coordinate system."""
        x, y = point
        h, w = persp_shape
        pano_h, pano_w = pano_shape
        
        fov = np.deg2rad(fov_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        # Camera parameters (matching your implementation)
        fx = w / (2 * np.tan(fov / 2))
        fy = fx
        cx = w / 2
        cy = h / 2
        
        # Convert pixel to normalized camera coordinates (matching your NDC)
        cam_x = (x - cx) / fx
        cam_y = -(y - cy) / fy  # Note: matching your -y
        cam_z = 1.0
        
        # Ray direction in camera frame (matching your vec calculation)
        vec_cam = np.array([cam_x, cam_y, cam_z])
        vec_cam = vec_cam / np.linalg.norm(vec_cam)
        
        # Transform to world coordinates (matching your rotation)
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                    [0, 1, 0],
                    [np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
        R = Ry @ Rx
        vec_world = R @ vec_cam
        
        # Convert to spherical coordinates
        lon = np.arctan2(vec_world[0], vec_world[2])
        lat = np.arcsin(np.clip(vec_world[1], -1, 1))
        
        # Convert to panorama pixel coordinates (matching your mapping)
        u = (lon / (2 * np.pi) + 0.5) * pano_w
        v = (0.5 - lat / np.pi) * pano_h
        
        # Handle longitude wrapping
        u = u % pano_w
        
        if 0 <= v < pano_h:
            return [u, v]
        else:
            return None

    # for visualization
    def _convert_panorama_bboxes_to_dict(self, panorama_bboxes):
        """
        Convert panorama_bboxes list to box_dict format for visualization.
        
        Args:
            panorama_bboxes: List of bbox dicts from reconstruct_panorama_bboxes
            
        Returns:
            dict: box_dict format compatible with visualize_bboxes
                Format: {'node_id': [...], 'annotation_bbox': [...], 'phrases': [...], 'confidence': [...]}}
        """
        if not panorama_bboxes:
            return {}
        
        # Extract arrays from panorama_bboxes
        node_ids = [bbox['node_id'] for bbox in panorama_bboxes]
        annotation_bboxes = [bbox['coords'] for bbox in panorama_bboxes]
        phrases = [bbox['phrase'] for bbox in panorama_bboxes]
        confidences = [bbox.get('confidence', 0.0) for bbox in panorama_bboxes]
        
        return {
                'node_id': node_ids,
                'annotation_bbox': annotation_bboxes,
                'phrases': phrases,
                'confidence': confidences
        }


# [TODO] =============================== BBOX TRACKING Functions ===============================
    def transform_panorama_bboxes_to_view(self, panorama_bboxes, fov_deg, num_views, out_shape=(1024, 1024), tilt_deg=0, tilt_views=None):
        """
        Transform panorama bounding boxes to perspective view coordinates for multiple views.
        Skips actual image extraction for efficiency when only bbox coordinates are needed.
        Args:
            panorama_bboxes: list of bbox info from reconstruct panorama bboxes only
            fov_deg: field of view in degrees
            num_views: number of views to generate
            out_shape: output perspective view shape (H, W)
        Returns:
            dict: bbox_dict mapping view filenames to bbox annotations
                Format: {base_filename: {'node_id': [B1, B2, ...], 'annotation_bbox': [[x1,y1,x2,y2], [x1,y1,x2,y2], ...], 'phrases': [P1, P2, ...], 'confidence': [C1, C2, ...]}
                Or {base_filename: {'node_id': [B1, B2, ...], 'annotation_bbox': [[[x,y], [x,y], ...], [[x,y], [x,y], ...], ...], 'phrases': [P1, P2, ...], 'confidence': [C1, C2, ...]}
        """
        # Handle empty panorama_bboxes gracefully
        if not panorama_bboxes:
            logging.warning("No panorama bboxes provided, returning empty bbox_dict")
            return {}
        
        view_filename = panorama_bboxes[0]['source_view']
        params = self.parse_perspective_filename(view_filename)
        filename_prefix = params['base_name']
        pano_shape = params['shape']

        # Generate yaw angles (clockwise)
        yaw_angles = np.linspace(0, 360, num_views, endpoint=False)[::-1]
        bbox_dict = {}
        
        for i, yaw in enumerate(yaw_angles):
            # Determine pitch angle for this view
            pitch_deg = tilt_views.get(i, tilt_deg) if tilt_views else tilt_deg
            
            # Generate view filename
            view_filename = f"{filename_prefix}@{fov_deg}@{yaw:.1f}@{pitch_deg:+.1f}@{pano_shape[0]}x{pano_shape[1]}.jpg"
            
            # Transform bboxes directly inline
            view_bboxes = []
            for bbox_info in panorama_bboxes:
                # Transform bbox coordinates
                persp_coords = self._transform_bbox_pano2persp(
                    bbox_info['coords'], fov_deg, yaw, pitch_deg, 
                    pano_shape, out_shape
                )
                
                if persp_coords is not None:
                    view_bboxes.append({
                        'node_id': bbox_info['node_id'],
                        'phrase': bbox_info['phrase'],
                        'annotation_bbox': persp_coords,
                        'confidence': bbox_info.get('confidence', 0.0)
                    })
            
            if view_bboxes:
                bbox_dict[view_filename] = {
                    'node_id': [bbox['node_id'] for bbox in view_bboxes],
                    'annotation_bbox': [bbox['annotation_bbox'] for bbox in view_bboxes],
                    'phrases': [bbox['phrase'] for bbox in view_bboxes],
                    'confidence': [bbox.get('confidence', 0.0) for bbox in view_bboxes]
                }
            else:
                logging.debug(f"No bboxes found for view: {view_filename}")
        
        return bbox_dict

    def _transform_bbox_pano2persp(self, pano_coords, fov_deg, yaw_deg, pitch_deg, pano_shape, persp_shape):
        """Safe bbox transformation using shared mapping logic"""
        if not pano_coords:
            return None
        
        # Analytically project each panorama point into the perspective view
        projected_points = []
        for u_target, v_target in pano_coords:
            pt = self._project_pano_point_to_persp(
                u_target, v_target, fov_deg, yaw_deg, pitch_deg, pano_shape, persp_shape
            )
            if pt is not None:
                projected_points.append(pt)
        
        if not projected_points:
            return None
        
        # Clip the polygon to the perspective view rectangle to handle partial visibility
        h, w = persp_shape
        clipped_polygon = self._clip_polygon_to_rect(projected_points, 0, 0, w - 1, h - 1)
        
        return clipped_polygon if clipped_polygon else None

    def _project_pano_point_to_persp(self, u, v, fov_deg, yaw_deg, pitch_deg, pano_shape, persp_shape):
        """
        Project a single panorama pixel coordinate (u, v) into perspective view coordinates (x, y).
        Returns None if the corresponding ray is behind the camera.
        """
        pano_h, pano_w = pano_shape
        persp_h, persp_w = persp_shape
        
        # Convert pano pixel to spherical lon/lat (radians)
        lon = (u / pano_w - 0.5) * 2 * np.pi
        lat = (0.5 - v / pano_h) * np.pi
        
        # World direction vector
        xw = np.cos(lat) * np.sin(lon)
        yw = np.sin(lat)
        zw = np.cos(lat) * np.cos(lon)
        vec_world = np.array([xw, yw, zw])
        
        # Camera rotation (consistent with other functions)
        fov = np.deg2rad(fov_deg)
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                       [0, 1, 0],
                       [np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        R = Ry @ Rx
        
        # Transform world ray to camera coordinates
        vec_cam = R.T @ vec_world
        if vec_cam[2] <= 0:
            return None  # Behind camera
        
        # Camera intrinsics (horizontal FOV)
        fx = fy = persp_w / (2 * np.tan(fov / 2))
        cx, cy = persp_w / 2.0, persp_h / 2.0
        
        x_img = vec_cam[0] / vec_cam[2] * fx + cx
        y_img = -vec_cam[1] / vec_cam[2] * fy + cy
        
        return [float(x_img), float(y_img)]

    def _clip_polygon_to_rect(self, polygon, x_min, y_min, x_max, y_max):
        """
        Clip a polygon to an axis-aligned rectangle using the Sutherland–Hodgman algorithm.
        polygon: list of [x, y]
        Returns a list of [x, y] or empty list if fully outside.
        """
        def inside(p, edge):
            x, y = p
            if edge == 'left':
                return x >= x_min
            if edge == 'right':
                return x <= x_max
            if edge == 'top':
                return y >= y_min
            if edge == 'bottom':
                return y <= y_max
            return True
        
        def intersect(p1, p2, edge):
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            if edge == 'left':
                x = x_min
                if dx == 0:
                    return [x_min, y1]
                t = (x_min - x1) / dx
                y = y1 + t * dy
                return [x, y]
            if edge == 'right':
                x = x_max
                if dx == 0:
                    return [x_max, y1]
                t = (x_max - x1) / dx
                y = y1 + t * dy
                return [x, y]
            if edge == 'top':
                y = y_min
                if dy == 0:
                    return [x1, y_min]
                t = (y_min - y1) / dy
                x = x1 + t * dx
                return [x, y]
            if edge == 'bottom':
                y = y_max
                if dy == 0:
                    return [x1, y_max]
                t = (y_max - y1) / dy
                x = x1 + t * dx
                return [x, y]
            return [x1, y1]
        
        def clip_with_edge(subject, edge):
            if not subject:
                return []
            output = []
            s = subject[-1]
            for e in subject:
                if inside(e, edge):
                    if inside(s, edge):
                        output.append(e)
                    else:
                        output.append(intersect(s, e, edge))
                        output.append(e)
                else:
                    if inside(s, edge):
                        output.append(intersect(s, e, edge))
                s = e
            return output
        
        # Ensure polygon is closed for clipping stability
        subject = polygon[:]
        
        for edge in ['left', 'right', 'top', 'bottom']:
            subject = clip_with_edge(subject, edge)
            if not subject:
                break
        
        # Deduplicate near-identical consecutive points
        def dedup(points, eps=1e-3):
            if not points:
                return points
            out = [points[0]]
            for p in points[1:]:
                if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
                    out.append(p)
            if len(out) > 1 and abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
                out.pop()
            return out
        
        return dedup(subject)

# ===== UTILITY FUNCTIONS =============================================================
  
    def parse_perspective_filename(self, filepath_or_name):
        """
        Parse parameters from encoded filename.
        Args:
            filepath: filename = f"{base_name}@{fov_deg}@{yaw_deg:.1f}@{pitch_deg:+.1f}@{out_shape[0]}x{out_shape[1]}.jpg"
            base_name: 37.7660763_-122.442508738@2014-08 or only 37.7660763_-122.442508738 or 1 ST_2%37.7660763_-122.442508738@2014-08
        
        Returns:
            dict: {'base_name', 'fov_deg', 'yaw_deg', 'pitch_deg', 'shape'}
        """
        filename = os.path.splitext(os.path.basename(filepath_or_name))[0]
        
        # Use rsplit to handle base_name that might contain @ symbols
        # Split from the right to get the last 4 parts: fov, yaw, pitch, shape
        parts = filename.rsplit('@', 4)
        
        if len(parts) != 5:
            raise ValueError(f"Invalid filename format: {filename}")
        
        base_name, fov, yaw, pitch, shape = parts
        height, width = map(int, shape.split('x'))
        
        return {
            'base_name': base_name,
            'fov_deg': float(fov),
            'yaw_deg': float(yaw),
            'pitch_deg': float(pitch),
            'shape': (height, width)
        }   

    def _get_perspective_mapping(self, fov_deg, yaw_deg, pitch_deg, pano_shape, persp_shape):
        """
        Generate the coordinate mapping used by _helper_equirect2persp.
        Returns the mapping arrays that can be reused.
        """
        w, h = persp_shape
        fov = np.deg2rad(fov_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        
        # Camera parameters
        fx = w / (2 * np.tan(fov / 2))
        fy = fx
        cx = w / 2
        cy = h / 2

        # Pixel grid
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        x = (xs - cx) / fx
        y = (ys - cy) / fy
        z = np.ones_like(x)

        # Ray directions in the camera frame
        vec = np.stack([x, -y, z], axis=2)
        norm = np.linalg.norm(vec, axis=2, keepdims=True)
        vec = vec / norm
        
        # Camera rotation
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                    [0, 1, 0],
                    [np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
        R = Ry @ Rx

        # Rotate vectors
        vec_cam = vec @ R.T

        # Spherical coordinates
        lon = np.arctan2(vec_cam[..., 0], vec_cam[..., 2])
        lat = np.arcsin(vec_cam[..., 1])

        # Map to panorama pixel coordinates
        pano_h, pano_w = pano_shape
        uf = (lon / (2 * np.pi) + 0.5) * pano_w
        vf = (0.5 - lat / np.pi) * pano_h
        
        return uf, vf

# =============================== Visualization Functions ===============================

    def visualize_bboxes(self, image_path, bbox_data, color=(0, 0, 255), thickness=2):
        """
        Elegantly draw bounding boxes on image.
        
        Args:
            image_path: Input image path
            bbox_data: Either:
                - Dict with keys: {'node_id': [...], 'annotation_bbox': [...], 'phrases': [...], 'confidence': [...]}
                - List of individual bbox dicts (legacy support)
            color: Drawing color (BGR tuple)
            thickness: Line thickness
        
        Returns:
            numpy.ndarray: Image with bboxes drawn (BGR format)
        """
        image = cv2.imread(image_path)
        result = image.copy()
        
        # Handle the actual format: {node_id: [...], annotation_bbox: [...], phrases: [...]}
        if isinstance(bbox_data, dict) and 'annotation_bbox' in bbox_data:
            node_ids = bbox_data.get('node_id', [])
            annotation_bboxes = bbox_data.get('annotation_bbox', [])
            phrases = bbox_data.get('phrases', [])
            confidences = bbox_data.get('confidence', [])
            
            # Ensure all lists have the same length
            max_len = max(len(node_ids), len(annotation_bboxes), len(phrases), len(confidences)) if any([node_ids, annotation_bboxes, phrases, confidences]) else 0
            
            for i in range(max_len):
                # Safely get values with defaults
                node_id = node_ids[i] if i < len(node_ids) else None
                bbox_coords = annotation_bboxes[i] if i < len(annotation_bboxes) else None
                phrase = phrases[i] if i < len(phrases) else None
                confidence = confidences[i] if i < len(confidences) else None
                
                if bbox_coords is not None:
                    result = self._draw_single_bbox(result, bbox_coords, color, thickness, 
                                                node_id, phrase, confidence)
        
        # Legacy support: list of individual bbox dicts
        elif isinstance(bbox_data, list):
            for bbox_info in bbox_data:
                if 'annotation_bbox' in bbox_info:
                    result = self._draw_single_bbox(
                        result, 
                        bbox_info['annotation_bbox'], 
                        color, 
                        thickness,
                        bbox_info.get('node_id'),
                        bbox_info.get('phrase'),
                        bbox_info.get('confidence')
                    )
        
        return result

    def _draw_single_bbox(self, image, bbox_coords, color, thickness, node_id=None, phrase=None, confidence=None):
        """
        Draw a single bounding box on the image.
        Handles both [x1,y1,x2,y2] and [[x,y], [x,y], ...] formats.
        """
        if bbox_coords is None:
            return image
        
        # Detect and handle different bbox formats
        if self._is_standard_bbox_format(bbox_coords):
            # Standard [x1, y1, x2, y2] format
            x1, y1, x2, y2 = map(int, bbox_coords)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label_pos = (x1, max(20, y1 - 10))  # Ensure y >= 20 for label visibility
            
        elif self._is_coords_format(bbox_coords):
            # Coordinate list [[x,y], [x,y], ...] format
            points = np.array([[int(x), int(y)] for x, y in bbox_coords], dtype=np.int32)
            
            if len(points) >= 3:
                # Draw polygon
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            elif len(points) == 2:
                # Draw line
                cv2.line(image, tuple(points[0]), tuple(points[1]), color, thickness)
            
            # Draw individual points
            for point in points:
                cv2.circle(image, tuple(point), max(1, thickness), color, -1)
            
            label_pos = tuple(points[0]) if len(points) > 0 else (10, 10)
            label_pos = (label_pos[0], max(20, label_pos[1] - 10))  # Ensure y >= 20 for label visibility
        
        else:
            return image  # Unknown format, skip
        
        # Add label
        label_parts = []
        if node_id is not None:
            label_parts.append(str(node_id))
        if phrase:
            label_parts.append(f"({phrase})")
        if confidence is not None:
            label_parts.append(f"{confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(image, label, label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return image

    def _is_standard_bbox_format(self, bbox_data):
        """
        Check if bbox_data is in standard [x1, y1, x2, y2] format.
        """
        return (isinstance(bbox_data, (list, tuple)) and 
                len(bbox_data) == 4 and 
                all(isinstance(x, (int, float)) for x in bbox_data))

    def _is_coords_format(self, bbox_data):
        """
        Check if bbox_data is in coordinate list [[x,y], [x,y], ...] format.
        """
        return (isinstance(bbox_data, list) and 
                len(bbox_data) > 0 and 
                all(isinstance(coord, (list, tuple)) and len(coord) == 2 
                    and all(isinstance(x, (int, float)) for x in coord) 
                    for coord in bbox_data))
    
    def save_bbox_annotations(self, bboxes, filepath):
        """
        Save bbox annotations to JSON file.
        
        Args:
            bboxes: bbox_dict mapping filename to bbox data dict
                   Format: {filename: {'node_id': [...], 'annotation_bbox': [...], 'phrases': [...], 'confidence': [...]}}
            filepath: Output JSON file path
        """
        if not isinstance(bboxes, dict):
            raise ValueError(f"Expected dict, got {type(bboxes)}")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Convert the entire bbox_dict
        serializable_bboxes = convert_numpy_types(bboxes)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_bboxes, f, indent=2)

    def reconstruct_boxes_pipeline(self, dino_folder, bboxes_dict, output_dir, views_folder, fov_deg, num_views, out_shape, tilt_deg, tilt_views=None, annotate_views=False, pano_folder=None):
        """
        Reconstruct boxes for all views in the dino_folder.
        Args:
            dino_folder: folder containing dino annotated bboxes
            bboxes_dict: bbox_dict mapping filename to bbox data dict
            output_dir: output directory for the reconstructed-extracted bboxes
            views_folder: folder containing all view images for the newly extracted views
            fov_deg: field of view degree
            num_views: number of views
            out_shape: output shape
            tilt_deg: tilt degree
            tilt_views: tilt views
            annotate_views: whether to annotate the views
        """
        from tqdm import tqdm
        
        view_files = glob.glob(os.path.join(dino_folder, "*.jpg"))
        bbox_dicts = {}

        if pano_folder is not None:
            # copy all pano images to output_dir for continous annotation
            for pano_filename in os.listdir(pano_folder):
                shutil.copy(os.path.join(pano_folder, pano_filename), os.path.join(output_dir, pano_filename))

        # Add progress bar for processing view files
        processed_count = 0
        skipped_count = 0
        
        for view_file_path in tqdm(view_files, desc="Reconstructing boxes", unit="view"):
            view_filename = os.path.basename(view_file_path)
            persp_shape = cv2.imread(view_file_path).shape[:2]
        
            # 1. Reconstruct panorama with bbox tracking
            if view_filename not in bboxes_dict:
                logging.warning(f"No bbox data found for {view_filename}, skipping")
                skipped_count += 1
                continue
                
            bboxes_dict_used = bboxes_dict[view_filename]
            pano_bboxes = self.reconstruct_panorama_bboxes(view_filename, bboxes_dict_used, persp_shape=persp_shape)

            # Skip processing if no panorama bboxes were reconstructed
            if not pano_bboxes:
                logging.debug(f"No panorama bboxes reconstructed for {view_filename}, skipping")
                skipped_count += 1
                continue

            # [TODO] only for debug use, remove later
            if pano_folder is not None:
                # Use the copied pano image for annotation
                pano_filename =  self.parse_perspective_filename(view_filename)['base_name']+".jpg"
                pano_path = os.path.join(output_dir, pano_filename)
                # use ['coords'] to draw bboxes on the panorama
                pano_bbox_dict = self._convert_panorama_bboxes_to_dict(pano_bboxes)
                annotated_pano = self.visualize_bboxes(pano_path, pano_bbox_dict, thickness=16)
                cv2.imwrite(os.path.join(output_dir, pano_filename), annotated_pano, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        
            # 2. Extract new views with automatically tracked bboxes
            bbox_dict = self.transform_panorama_bboxes_to_view(
                panorama_bboxes=pano_bboxes, fov_deg=fov_deg, num_views=num_views, out_shape=out_shape, tilt_deg=tilt_deg, tilt_views=tilt_views)

            bbox_dicts.update(bbox_dict)
            processed_count += 1
            
            # 4. Visualize the bboxes
            if annotate_views:
                # 5. Visualize the bboxes
                for view_filename, bbox_data in bbox_dict.items():
                    view_path = os.path.join(views_folder, view_filename)
                    view_img = self.visualize_bboxes(view_path, bbox_data)
                    # save_path = os.path.join(output_dir, view_filename)
                    cv2.imwrite(view_path, view_img, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]) # overwrite the original view image so all the bboxes can be annotated on same image

        # 3. Save the collected bbox_dict
        bbox_dict_path = os.path.join(output_dir, "extracted_bboxes.json")
        self.save_bbox_annotations(bbox_dicts, bbox_dict_path)
        
        # Print summary statistics
        print(f"\n✅ Box reconstruction pipeline complete!")
        print(f"   - Total files processed: {len(view_files)}")
        print(f"   - Successfully processed: {processed_count}")
        print(f"   - Skipped (no bbox data): {skipped_count}")
        print(f"   - Output saved to: {bbox_dict_path}")

def run_resize_image_pipeline():
    processor = ImgProcessor()
    img_folder = "D:/VPR_dataset/tokyo247/247query_subset_v2_extractedUTM"
    resized_folder = img_folder + "_resized"
    processor.resize_image_pipeline(img_folder, resized_folder, pixel_max=1024)

def run_pano_pipeline():
    pano_folder = "F:/VPR_dataset/hk-urban/gsv_pano"
    extracted_folder = pano_folder + "_extracted"
    os.makedirs(extracted_folder, exist_ok=True)
    print("Extracted folder: ", extracted_folder)
    processor = ImgProcessor()
    
    # Step 1: Convert panorama to perspective views
    print("Converting panorama to perspective views")
    processor.convert_panorama_batch_pipeline(pano_folder, extracted_folder, 60, 12, out_shape=(640, 480), tilt_deg=-12, max_workers=20)
    # processor.convert_panorama_batch_pipeline(pano_folder, extracted_folder, 110, 4, out_shape=(1024, 1024), tilt_deg=0, tilt_views={0: -15, 1: -30, 2: -15, 3: 30, 4: -15, 5: -30, 6: -15, 7: 30}, max_workers=10)
    
    # Step 2: Separate perspective views to subfolders
    # print("Separating perspective views to subfolders")
    # processor.separate_perspective_subfolders_pipeline(extracted_folder, yaw_dict={"folder_1": [90, 270], "folder_2": [0, 180]})

    # # Step 3: CALL GROUNDDINO TO GET BBOX AND TEXT
    # print("Calling GroundDino to get bbox and text")
    # schema_dict = {'B': ['single building'], 'V': ['single vegetation']}
    # dino = DinoAnnotator(box_threshold=0.3, text_threshold=0.2, logging_level=logging.ERROR)
    
    # # Example 2: Batch processing (uncomment to use)
    # image_folder = os.path.join(extracted_folder, "folder_1")
    # print("Calling GroundDino for processing image_folder: ", image_folder, "using schema_dict: ", schema_dict)
    # batch_results = dino.process_dino_schema_batch(
    #     img_folder=image_folder,
    #     schema_dict=schema_dict,
    #     max_num=4,
    #     save_annotated=True,
    #     max_workers=12
    # )
    # print(f"Batch processing completed: {len(batch_results)} images processed")

    # Step 3.2: Merge perspective views to parent folder
    # processor.merge_perspective_subfolders(extracted_folder)

    # Step 4: Reconstruct panorama from dino annotated perspective views
    # processor.batch_reconstruct_panoramas(
    #     extracted_folder, 
    #     os.path.join(working_folder, "target_pano_2022_1186places_dinoreconstructed"),
    #     max_pano_size=16384, # 4096 for quick test, 16384 for full panorama
    #     memory_limit_gb=4,
    #     max_workers=8  # Use parallel processing for faster reconstruction
    # )

    # Step 5: Convert reconstructed panorama to perspective views
    # reconstructed_folder = os.path.join(working_folder, "panorama_reconstructed")
    # reconstructed_extracted_folder = os.path.join(working_folder, "panorama_reconstructed_extracted")
    # processor.convert_panorama_batch_pipeline(reconstructed_folder, reconstructed_extracted_folder, 120, 4, out_shape=(1024, 1024), tilt_deg=0, max_workers=10)

def run_pano_box_reconstruction():
    processor = ImgProcessor() 
    pano_folder = "D:/git_projects/LLM_VPR/dataset_multimodal/test_embedding/target_pano"

    working_folder = os.path.join(os.getcwd(), "dataset_multimodal", "test_embedding")
    extracted_folder = os.path.join(working_folder, "target_bboxDINO")

    bbox_views_folder = os.path.join(extracted_folder, "pano_bbox_views") # full extracted folder, containing all views  
    os.makedirs(bbox_views_folder, exist_ok=True)

    # configure pano2persp paramas
    fov_deg=90
    num_views=8
    out_shape=(1024, 1024)
    tilt_deg=0
    tilt_views=None

    # # Step 1: Convert panorama to perspective views for all bboxes views
    # print("Converting panorama to perspective views for all future bboxes views")
    # processor.convert_panorama_batch_pipeline(pano_folder, bbox_views_folder, fov_deg, num_views, out_shape, tilt_deg, tilt_views, max_workers=20)

    # # 2. Reconstruct pano bboxes for all views in the dino_folder
    dino_folder = os.path.join(working_folder, "target_extracted1_dino") # folder containing dino annotated bboxes
    bbox_json_file = os.path.join(dino_folder, "bbox_dict", "dino_map.json")
    bboxes_dict = json.load(open(bbox_json_file, "r"))

    output_dir = os.path.join(extracted_folder, "output_bbox")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reconstructing pano bboxes for all views in the dino_folder")
    processor.reconstruct_boxes_pipeline(dino_folder, bboxes_dict, output_dir, bbox_views_folder, fov_deg, num_views, out_shape, tilt_deg, tilt_views, annotate_views=False)
    
if __name__ == "__main__":
    # run_resize_image_pipeline()
    run_pano_pipeline()
    # run_pano_box_reconstruction()
    

    # processer = ImgProcessor(log_level=logging.ERROR)
    # img_folder = os.path.join(os.getcwd(), "dataset_multimodal", "query100@target1186", "query_100")
    # resized_folder = img_folder + "_resized"
    # processer.resize_image_pipeline(img_folder, resized_folder, pixel_max=1024)

    """ process mapillary folder for read-to-use query images """
    # mapillary_folder = os.path.join(working_dir, "dataset_svi", "sf_query_mapillary158")
    # processor.process_mapillary_panorama(mapillary_folder)
