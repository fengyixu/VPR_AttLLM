import os
from PIL import Image
from pathlib import Path
from batch_controller import BatchController

import logging
logging.basicConfig(level=logging.INFO)


class ImgProcessor:
    def __init__(self):
        pass

    def resize_image(self, image_path, pixel_max=1024):
        """
        Resize image's longer side to pixel_max while maintaining aspect ratio.

        Args:
            image_path (str): Path to the input image
            pixel_max (int): Maximum pixels for the longer side (default: 1024)

        Returns:
            PIL.Image: Resized image object
        """
        img = Image.open(image_path)
        width, height = img.size

        if width > height:
            scale_factor = pixel_max / width
        else:
            scale_factor = pixel_max / height

        if scale_factor < 1:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background

        return img

    def resize_and_save(self, image_path, output_folder, pixel_max=1024):
        """
        Resize image and save to output_folder.

        Args:
            image_path (str): Path to the input image
            output_folder (str): Folder to save the resized image
            pixel_max (int): Maximum pixels for the longer side

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
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            Path(os.path.join(image_folder, f))
            for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f))
            and Path(f).suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No image files found in {image_folder}")
            return

        def process_image(image_file):
            item_key = image_file.name
            try:
                output_path = self.resize_and_save(str(image_file), output_folder, pixel_max)
                return {"status": "success", "file": item_key, "output": os.path.basename(output_path)}
            except Exception as e:
                return {"status": "error", "file": item_key, "error": str(e)}

        print(f"Starting parallel image resizing:")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total images to process: {len(image_files)}")
        print(f"   - Target pixel max: {pixel_max}")

        batch_controller = BatchController(
            save_dir=os.path.join(output_folder, "checkpoint"),
            checkpoint_interval=10
        )
        batch_controller.run_batch_parallel(
            items=image_files,
            process_func=process_image,
            common_params={},
            resume=True,
            desc="Resizing images in parallel",
            max_workers=max_workers
        )

        successful = sum(1 for r in batch_controller.results if r.get("status") == "success")
        failed = len(batch_controller.results) - successful

        print(f"\nImage resizing complete!")
        print(f"   - Successfully processed: {successful} images")
        print(f"   - Failed to process: {failed} images")
        print(f"   - Output saved to: {output_folder}")
