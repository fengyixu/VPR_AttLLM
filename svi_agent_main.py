import os
import logging
import json
import re
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
import pandas as pd
import shutil
from tqdm import tqdm
from agent_bot import GeminiAgent
from prompts import PromptManager
# from agent_bot import QwenAgent, GeminiAgent,RateController
# from svi_graph import GraphBuilder
from svi_preprocess import ImgProcessor
from batch_controller import BatchController
from svi_json_clean import JsonDataCleaner
import threading
import matplotlib
# Set non-interactive backend to avoid threading issues
matplotlib.use('Agg')  # Must be set before importing pyplot
from matplotlib import pyplot as plt
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
# Suppress only the google_genai.models logger
logging.getLogger('google_genai.models').setLevel(logging.ERROR)
class SviAgent:

    def __init__(self, model_name, bot_class, bot_kwargs=None, max_qpm: int = 60, logger = logger):
        self.logger = logger
        self.model_name = model_name
        self.bot_class = bot_class
        self.bot_kwargs = bot_kwargs or {}
        self.prompt_manager = PromptManager()
        self.img_processor = ImgProcessor()
        # self.graph_builder = GraphBuilder(flattern_json=None, debug=False, true_graph=None)
        self.cleaner = JsonDataCleaner()
        # self.rate_controller = RateController(max_qpm=max_qpm)  # Remove usage for now
        # Thread lock for matplotlib operations to ensure thread safety
        self._matplotlib_lock = threading.Lock()

# ================================= CORE FUNCTIONS =================================
    def analyze_image(self, bot, prompt_list: list[str], image_path: str, save_folder: str = None, preprocess_img: bool = False, grid_num=4, grid_line=False, axis=False):
        """
        Analyze a single image using multiple prompts sequentially.
        
        Args:
            bot: Bot instance for image analysis
            prompt_list: List of prompts to process sequentially
            image_path: Path to the image file (single image only)
            save_folder: Folder to save JSON results
            llm_clean: Whether to use LLM for JSON cleaning
            preprocess_img: If True, overlay outside axes before sending to the bot
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        if not image_path:
            self.logger.error("Please provide an image path")
            return None
        if len(prompt_list) == 0:
            self.logger.error("Please provide a list of prompts to test by sequence")
            return None

        try:
            # Use chat_image to process all prompts sequentially
            img_or_path = self.preprocess_image(image_path, grid_num=grid_num, grid_line=grid_line, axis=axis) if preprocess_img else image_path
            responses = bot.chat_image(image_path=img_or_path, prompt_list=prompt_list)
            
            # Get the last response (as per requirement)
            final_response = responses[-1] if responses else ""
            
        except Exception as e:
            self.logger.error(f"[API CALL] Failed to process image {image_path}: {str(e)}")
            raise RuntimeError(f"Image analysis failed: {str(e)}")
        finally:
            # Clear the conversation after processing
            bot.clear_conversation()
        # Process, verify and persist the response
        return self.check_save_response(final_response, image_path, bot.model, save_folder, grid_num=grid_num)

    def check_save_response(self, final_response: str, image_path: str, model_name: str, save_folder: Optional[str], grid_num=4) -> Dict[str, Any]:
        os.makedirs(save_folder or os.getcwd(), exist_ok=True)
        if grid_num > 0:
            check = self.cleaner.process_response_json(final_response, grid_num=grid_num)
        elif grid_num == 0:
            check = self.cleaner.process_response_list(final_response)
        else:
            raise ValueError(f"Invalid grid_num: {grid_num}")
        cleaned_json = check['cleaned_json']
        json_format_verified = check['json_format_verified']
        json_content_verified = check['json_content_verified']
        no_weighting = check.get('no_weighting', False)
        json_data = {
            "image_path": image_path,
            "model": model_name,
            "json_format_verified": bool(json_format_verified),
            "json_content_verified": bool(json_content_verified),
            # Persist explicit control: if no_weighting, store result as None and flag it
            "no_weighting": bool(no_weighting),
            "result": (None if no_weighting else (cleaned_json if (cleaned_json is not None and json_format_verified and json_content_verified) else final_response))
        }
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        save_path = os.path.join(save_folder or os.getcwd(), base_name + '.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved JSON to {save_path}")
        return json_data

    def preprocess_image(self, image_path: str, max_size: int = 1024, grid_num: int = 4, grid_line: bool = False, axis: bool = False) -> Image.Image:
        """Overlay outside rulers on all four sides with configurable grid ticks.
        
        Args:
            image_path: Path to the input image
            max_size: Maximum size for image resizing
            grid_num: Number of grid divisions (creates grid_num+1 ticks from 0.0 to 1.0)
            grid_line: If True, draw white grid lines with 2 pixel width
            axis: If True, draw axis with ticks; if False, hide axis
            
        Returns:
            PIL Image with overlaid rulers identical to plt.savefig output
        """
        # Use thread lock to ensure matplotlib operations are thread-safe
        with self._matplotlib_lock:
            if grid_num == 0:
                grid_num = 5 # use default tick with 0.2 interval
                grid_line = False # no grid line when grid_num is 0
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            scale = min(1.0, max_size / max(width, height))
            new_w, new_h = int(round(width * scale)), int(round(height * scale))
            if (new_w, new_h) != (width, height):
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Adaptive styling
            max_dim = max(new_w, new_h)
            scale_factor = max_dim / 1024
            label_size = int(round(7 + 3 * scale_factor))
            tick_len = int(round(3 + 3 * scale_factor))
            tick_width = 0.8 + 0.7 * scale_factor
            spine_width = 1.0 + 0.8 * scale_factor
            
            # Use higher DPI for LLM clarity
            dpi = 150
            label_px = label_size * dpi / 72.0
            h_margin = label_px * 2.5 + tick_len + 12
            v_margin = label_px + tick_len + 12
            
            total_w, total_h = new_w + 2 * h_margin, new_h + 2 * v_margin
            fig = plt.figure(figsize=(total_w/dpi, total_h/dpi), dpi=dpi)
            
            ax = fig.add_axes([h_margin/total_w, v_margin/total_h, 
                            new_w/total_w, new_h/total_h])
            
            ax.imshow(img, extent=[0, 1, 1, 0])
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect(new_h / new_w)
            
            # Generate ticks based on grid_num parameter
            tick_interval = 1.0 / grid_num
            ticks = [i * tick_interval for i in range(grid_num + 1)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            
            # Draw grid lines if requested (always set ticks first for grid to work)
            if grid_line:
                ax.grid(True, color='white', linewidth=2, alpha=1.0)
            
            # Control axis visibility
            if axis:
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
                
                ax.tick_params(direction='out', length=tick_len, width=tick_width,
                            labelsize=label_size, pad=3, which='major',
                            top=True, bottom=True, left=True, right=True,
                            labeltop=True, labelright=True)
            else:
                # Hide axis spines and tick labels, but keep tick positions for grid
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Hide tick labels and marks without removing tick positions
                ax.tick_params(direction='out', length=0, width=0,
                            pad=0, which='major',
                            top=False, bottom=False, left=False, right=False,
                            labeltop=False, labelright=False, labelbottom=False, labelleft=False)
                
                # Hide tick labels by setting them to empty strings (avoids fontsize warning)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            # Save to memory buffer using same method as plt.savefig
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close(fig)
            
            buf.seek(0)
            return Image.open(buf).convert('RGB')

    def generate_image(self, bot, prompt_list: list[str], image_path: list[str] = None, save_image: bool = False, save_folder: str = None):
        """
        Generate images based on prompts and optional reference images.
        
        Args:
            bot: Bot instance for image generation
            prompt_list: List of prompts for image generation
            image_path: Optional list of reference image paths
            save_image: Whether to save generated images
            save_folder: Folder to save generated images
            
        Returns:
            Dictionary containing generation results and metadata
        """
        if len(prompt_list) == 0:
            self.logger.error("Please provide a list of prompts for image generation")
            return None

        self.logger.debug(f"[IMAGE GEN] Starting image generation with {len(prompt_list)} prompts")
        
        # Count tokens for the first prompt
        token_info = bot.count_tokens(prompt_list[0], image_path)
        self.logger.debug(f"[TOKEN COUNT] Prompt tokens: {token_info.get('text_tokens', 0)}, Image tokens: {token_info.get('image_tokens', 0)}, Total: {token_info.get('total_tokens', 0)}")
        
        # Generate image using the first prompt
        generation_result = bot.image_gen(
            text_prompt=prompt_list[0],
            image_paths=image_path,
            temperature=0.7
        )
        
        self.logger.debug(f"[IMAGE GEN] Generated {len(generation_result.get('generated_images', []))} images")
        
        # Process additional prompts if provided
        if len(prompt_list) > 1:
            self.logger.debug(f"[IMAGE GEN] Processing {len(prompt_list) - 1} additional prompts")
            for i, prompt in enumerate(prompt_list[1:], 1):
                # Count tokens for each additional prompt
                additional_token_info = bot.count_tokens(prompt, image_path)
                self.logger.debug(f"[TOKEN COUNT] Additional prompt {i} tokens: {additional_token_info.get('text_tokens', 0)}, Total: {additional_token_info.get('total_tokens', 0)}")
                
                # Generate additional images
                additional_result = bot.image_gen(
                    text_prompt=prompt,
                    image_paths=image_path,
                    temperature=0.7
                )
                
                # Merge results
                generation_result['generated_images'].extend(additional_result.get('generated_images', []))
                generation_result['generated_text'] += "\n" + additional_result.get('generated_text', '')
                generation_result['metadata']['additional_prompts'] = generation_result['metadata'].get('additional_prompts', []) + [prompt]
        
        # Save generated images if requested
        if save_image and save_folder and generation_result.get('generated_images'):
            os.makedirs(save_folder, exist_ok=True)
            saved_paths = []
            
            for i, image in enumerate(generation_result['generated_images']):
                # Generate filename based on original image name if available
                if image_path and len(image_path) > 0:
                    base_name = os.path.splitext(os.path.basename(image_path[0]))[0]
                    filename = f"{base_name}_generated_{i+1}.png"
                else:
                    filename = f"generated_image_{i+1}.png"
                
                save_path = os.path.join(save_folder, filename)
                image.save(save_path)
                saved_paths.append(save_path)
                self.logger.info(f"[IMAGE GEN] Saved generated image to: {save_path}")
            
            generation_result['saved_paths'] = saved_paths
        
        return generation_result

    def _sync_checkpoint_with_outputs(self, items_to_process, output_folder, output_suffix, checkpoint_path):
        """
        Mark items as processed if a file with the same base name (plus output_suffix) exists in the output folder.
        For '.json' output_suffix, expects exact base name match with .json extension.
        For '_generated', expects any file starting with the base name and containing '_generated'.
        """
        processed = set()
        output_files = set(os.listdir(output_folder))
        for fname in items_to_process:
            base_name = os.path.splitext(fname)[0]
            if output_suffix == '.json':
                expected_file = base_name + '.json'
                if expected_file in output_files:
                    processed.add(fname)
            else:
                # Accept any file that starts with base_name and contains output_suffix
                if any(f.startswith(base_name) and output_suffix in f for f in output_files):
                    processed.add(fname)
        # Write or update checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_data = {"processed_items": list(processed)}
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        return processed

    def _sync_checkpoint_with_result_json(self, items_to_process, result_json_path, checkpoint_path):
        """
        Sync checkpoint by comparing query image filenames with keys in the consolidated result JSON.
        Marks an item processed if its filename (including extension) exists as a key in the JSON.
        """
        processed = set()
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                if isinstance(result_data, dict):
                    present_keys = set(result_data.keys())
                    for fname in items_to_process:
                        if fname in present_keys:
                            processed.add(fname)
            except Exception as e:
                self.logger.error(f"Failed to load existing result JSON for checkpoint sync: {e}")
        os.path.dirname(checkpoint_path) and os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_items": list(processed)}, f, indent=2)
        return processed


# ================================= PIPELINE: SVI Attention =================================
    def svi_attention_batch_pipeline(self, svi_folder, prompt_list, json_save_folder, preprocess_img=False, grid_num=4, grid_line=False, axis=False, max_workers=4, retry=False):
        """
        Generate SVI graphs for all images in the test folder with progress tracking.
        Automatically syncs checkpoint with actual output files.
        Output file naming: output files should start with the input base name, with any extension or suffix (e.g. foo.json, foo_result.json).
        """
        os.makedirs(json_save_folder, exist_ok=True)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        svi_path = Path(svi_folder)
        image_files = [f for f in svi_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        if not image_files:
            self.logger.info(f"No image files found in {svi_folder}")
            return
        print(f"Found {len(image_files)} images in {svi_folder}, preprocess_img={preprocess_img}, grid_num={grid_num}, grid_line={grid_line}, axis={axis}")
        print(f"If Preprocess_img is True, outside axes will be overlayed onto image before sending to the bot")
        if input("do you want to continue? (y/n): ") != "y":
            return
        filename_to_path = {f.name: f for f in image_files}
        items_to_process = list(filename_to_path.keys())
        checkpoint_path = os.path.join(json_save_folder, "checkpoint", "batch_checkpoint.json")
        processed = self._sync_checkpoint_with_outputs(
            items_to_process, json_save_folder, '.json', checkpoint_path
        )
        unprocessed_items = [item for item in items_to_process if item not in processed]
        if not unprocessed_items:
            print(f"All items already processed. Nothing to do.")
            return

        # Thread-safe usage accumulator across all parallel workers
        usage_lock = threading.Lock()
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "api_calls": 0}

        def process_image(filename):
            item_key = filename
            self.logger.debug(f"[PROCESS] Starting process_image for: {item_key}")
            try:
                image_file = filename_to_path[filename]
                bot = self.bot_class(model=self.model_name, **self.bot_kwargs)
                bot.clear_conversation()
                base_name = os.path.splitext(filename)[0]
                json_file_path = os.path.join(json_save_folder, base_name + '.json')
                if os.path.exists(json_file_path):
                    self.logger.info(f"[PROCESS] JSON file already exists for {item_key}, skipping processing")
                    return {"status": "skipped", "file": item_key, "reason": "file_exists"}
                response_data = self.analyze_image(bot, prompt_list, str(image_file), save_folder=json_save_folder, preprocess_img=preprocess_img, grid_num=grid_num, grid_line=grid_line, axis=axis)
                bot.clear_conversation()
                # Accumulate token usage from this worker's bot
                bot_usage = bot.get_usage_summary()
                with usage_lock:
                    for k in total_usage:
                        total_usage[k] += bot_usage.get(k, 0)
                result_data = response_data.get("result", None)
                no_weighting = bool(response_data.get("no_weighting", False))
                # Treat explicit no_weighting as a successful outcome
                if response_data is not None and (result_data is not None or no_weighting):
                    self.logger.debug(f"[PROCESS] Finished process_image for: {item_key}")
                    return {"status": "success", "file": item_key, "response": result_data, "no_weighting": no_weighting}
                else:
                    raise Exception("Failed to get result")
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"✗ Error processing {item_key}: {error_msg}")
                raise Exception(f"Failed to process {item_key}: {error_msg}")
        print(f"🚀 Starting SVI graph parallel batch pipeline")
        print(f"   - Model: {self.model_name}")
        print(f"   - Number of prompts per image: {len(prompt_list)}")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total images to process: {len(unprocessed_items)} (of {len(items_to_process)})")
        if retry:
            print(f"   - Retry mode: Will retry failed items from previous run")

        wall_start = time.perf_counter()
        batch_controller = BatchController(
            save_dir=os.path.join(json_save_folder, "checkpoint"),
            checkpoint_interval=4
        )
        batch_controller.run_batch_parallel(
            items=items_to_process,
            process_func=process_image,
            common_params={},
            resume=True,
            desc="Processing SVI images in parallel",
            max_workers=max_workers,
            rate_limiter=None,
            retry=retry
        )
        wall_time = time.perf_counter() - wall_start

        # Build and print usage summary
        summary_lines = [
            f"=== API Usage Summary ===",
            f"  Model:             {self.model_name}",
            f"  Images processed:  {len(unprocessed_items)}",
            f"  API calls:         {total_usage['api_calls']}",
            f"  Prompt tokens:     {total_usage['prompt_tokens']}",
            f"  Completion tokens: {total_usage['completion_tokens']}",
            *([ f"  Thinking tokens:   {total_usage['thinking_tokens']}"] if total_usage['thinking_tokens'] else []),
            f"  Total tokens:      {total_usage['total_tokens']}",
            f"  Wall time:         {wall_time:.1f}s",
            f"  Timestamp:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        print("\n" + "\n".join(summary_lines))

        # Save usage summary to text file
        usage_dir = os.path.join(json_save_folder, "usage_summary")
        os.makedirs(usage_dir, exist_ok=True)
        usage_file = os.path.join(usage_dir, f"usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(usage_file, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"✅ JSON generation complete. Output saved to: {json_save_folder}")
        print(f"   Usage summary saved to: {usage_file}")

    def svi_generation_batch_pipeline(self, svi_folder, prompt_list, image_save_folder, max_workers=4, retry=False):
        """
        Generate images for all images in the test folder with progress tracking.
        Automatically syncs checkpoint with actual output files.
        Output file naming: output files should start with the input base name, with any extension or suffix (e.g. foo_generated.png, foo.png).
        """
        os.makedirs(image_save_folder, exist_ok=True)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        svi_path = Path(svi_folder)
        image_files = [f for f in svi_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        if not image_files:
            print(f"No image files found in {svi_folder}")
            return
        filename_to_path = {f.name: f for f in image_files}
        items_to_process = list(filename_to_path.keys())
        checkpoint_path = os.path.join(image_save_folder, "checkpoint", "batch_checkpoint.json")
        # Sync checkpoint with actual output files (accept any file with _generated or matching base name)
        processed = self._sync_checkpoint_with_outputs(
            items_to_process, image_save_folder, '_generated', checkpoint_path
        )
        unprocessed_items = [item for item in items_to_process if item not in processed]
        if not unprocessed_items:
            print(f"All items already processed. Nothing to do.")
            return
        def process_image_generation(filename):
            item_key = filename
            logger.debug(f"[IMAGE GEN] Starting image generation for: {item_key}")
            try:
                image_file = filename_to_path[filename]
                bot = self.bot_class(model=self.model_name, **self.bot_kwargs)
                bot.clear_conversation()  # Ensure clean state before processing
                image_path = str(image_file)
                base_name = os.path.splitext(filename)[0]
                # Accept any file that starts with base_name and contains _generated
                already_generated = any(
                    f.startswith(base_name) and '_generated' in f
                    for f in os.listdir(image_save_folder)
                )
                if already_generated:
                    logger.info(f"[IMAGE GEN] Generated image already exists for {item_key}, skipping processing")
                    return {"status": "skipped", "file": item_key, "reason": "file_exists"}
                token_info = bot.count_tokens(prompt_list[0], [image_path])
                logger.debug(f"[TOKEN COUNT] {item_key} - Prompt tokens: {token_info.get('text_tokens', 0)}, Image tokens: {token_info.get('image_tokens', 0)}, Total: {token_info.get('total_tokens', 0)}")
                generation_result = self.generate_image(
                    bot, 
                    prompt_list, 
                    [image_path], 
                    save_image=True, 
                    save_folder=image_save_folder
                )
                bot.clear_conversation()  # Ensure clean state after processing
                if generation_result and generation_result.get('generated_images'):
                    logger.debug(f"[IMAGE GEN] Finished image generation for: {item_key}")
                    return {"status": "success", "file": item_key, "generated_count": len(generation_result['generated_images'])}
                else:
                    error_msg = "Failed to generate images"
                    logger.error(f"[IMAGE GEN] Failed to generate images for: {item_key}: {error_msg}")
                    raise Exception(error_msg)
            except Exception as e:
                error_msg = str(e)
                print(f"\u2717 Error generating image for {item_key}: {error_msg}")
                logger.debug(f"[IMAGE GEN] Error in process_image_generation for: {item_key}, error: {error_msg}")
                raise Exception(f"Failed to generate image for {item_key}: {error_msg}")
        print(f"🎨 Starting SVI image generation parallel batch pipeline")
        print(f"   - Model: {self.model_name}")
        print(f"   - Number of prompts per image: {len(prompt_list)}")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total images to process: {len(unprocessed_items)} (of {len(items_to_process)})")
        if retry:
            print(f"   - Retry mode: Will retry failed items from previous run")
        batch_controller = BatchController(
            save_dir=os.path.join(image_save_folder, "checkpoint"),
            checkpoint_interval=4
        )
        batch_controller.run_batch_parallel(
            items=items_to_process,  # Pass total items instead of unprocessed items
            process_func=process_image_generation,
            common_params={},
            resume=True,
            desc="Generating images in parallel",
            max_workers=max_workers,
            rate_limiter=None,
            retry=retry
        )
        print(f"✅ Image generation complete. Output saved to: {image_save_folder}")

# ================================= Use LLM to find the top 100 matches =================================
    def find_matches_top100(self, prompt_list, query_image_path, target_image_paths):
        """
        Find the top 100 matches for a query image from a list of target images.
        """
        if not query_image_path or not target_image_paths:
            self.logger.error("Please provide valid query image path and target image paths")
            return None
        if len(prompt_list) == 0:
            self.logger.error("Please provide a list of prompts to test by sequence")
            return None
        bot = self.bot_class(model=self.model_name, **self.bot_kwargs)
        self.logger.debug(f"using bot: {self.model_name}")
        try:
            # Use chat_image to process all prompts sequentially
            responses = bot.chat(
                text_prompt=prompt_list[0],
                image_paths=[query_image_path] + target_image_paths,
                preserve_image_order=True
            )
            # Get the last response (as per requirement)
            cleaned_json = self.cleaner.process_top100_json(responses)
            
        except Exception as e:
            self.logger.error(f"Failed to process image {query_image_path}: {str(e)}")
            raise RuntimeError(f"Image analysis failed: {str(e)}")
        finally:
            # Clear the conversation after processing
            bot.clear_conversation()
        
        return cleaned_json

    def svi_top100_batch_pipeline(self, prompt_list, query_svi_folder, target_svi_folder, cosplace_json_file, json_save_folder, max_workers=10, retry=False):
        """
        Parallel pipeline for LLM-based top-100 re-ranking for all queries in a folder.
        - Uses cosplace_json_file to retrieve candidate target paths per query
        - Syncs checkpoint using keys in the consolidated results JSON
        - Writes a consolidated LLMrerank_results.json safely with thread locking
        Returns the consolidated results JSON path.
        """
        os.makedirs(json_save_folder, exist_ok=True)
        results_path = os.path.join(json_save_folder, "LLMrerank_results.json")
        checkpoint_path = os.path.join(json_save_folder, "checkpoint", "batch_checkpoint.json")
        # Ensure consolidated results exists
        if not os.path.exists(results_path):
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
        # Load cosplace mapping
        with open(cosplace_json_file, 'r', encoding='utf-8') as f:
            cosplace_map = json.load(f)
        # Collect query images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        q_path = Path(query_svi_folder)
        image_files = [f for f in q_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        if not image_files:
            self.logger.info(f"No image files found in {query_svi_folder}")
            return results_path
        filename_to_path = {f.name: f for f in image_files}
        # Only process files present in cosplace map
        items_to_process = [f.name for f in image_files if f.name in cosplace_map]
        if not items_to_process:
            self.logger.info("No matching queries found in cosplace mapping; nothing to process.")
            return results_path
        # Sync checkpoint from consolidated results JSON keys
        processed = self._sync_checkpoint_with_result_json(items_to_process, results_path, checkpoint_path)
        unprocessed_items = [it for it in items_to_process if it not in processed]
        if not unprocessed_items:
            print("All items already processed. Nothing to do.")
            return results_path
        # Thread lock for safe writes
        results_lock = threading.Lock()
        def process_query(filename):
            item_key = filename
            self.logger.debug(f"[TOP100] Start processing: {item_key}")
            try:
                query_image_path = str(filename_to_path[filename])
                # Build target paths (limit to top 100)
                target_rel = cosplace_map[filename].get("target_path", [])
                target_svi_paths = [os.path.join(target_svi_folder, p) for p in target_rel[:100]]
                if not target_svi_paths:
                    raise Exception("No target paths found in cosplace map")
                # Call LLM ranking
                response = self.find_matches_top100(prompt_list, query_image_path, target_svi_paths)
                # Persist into consolidated results with thread safety
                with results_lock:
                    try:
                        with open(results_path, 'r', encoding='utf-8') as rf:
                            current = json.load(rf) or {}
                    except Exception:
                        current = {}
                    current[filename] = response
                    with open(results_path, 'w', encoding='utf-8') as wf:
                        json.dump(current, wf, indent=2)
                self.logger.debug(f"[TOP100] Finished: {item_key}")
                return {"status": "success", "file": item_key}
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"✗ Error processing {item_key}: {error_msg}")
                raise Exception(f"Failed to process {item_key}: {error_msg}")
        print("🚀 Starting SVI top-100 parallel batch pipeline")
        print(f"   - Model: {self.model_name}")
        print(f"   - Max parallel workers: {max_workers}")
        print(f"   - Total queries to process: {len(unprocessed_items)} (of {len(items_to_process)})")
        if retry:
            print("   - Retry mode: Will retry failed items from previous run")
        batch_controller = BatchController(
            save_dir=os.path.join(json_save_folder, "checkpoint"),
            checkpoint_interval=4
        )
        batch_controller.run_batch_parallel(
            items=items_to_process,
            process_func=process_query,
            common_params={},
            resume=True,
            desc="Processing SVI top-100 in parallel",
            max_workers=max_workers,
            rate_limiter=None,
            retry=retry
        )
        print(f"✅ Top-100 re-ranking complete. Results saved to: {results_path}")
        return results_path

def run_llm_find_matches_top100():
    # Initialize components
    MODEL_NAME = "gemini-2.5-flash"
    # Pass the class, not an instance
    svi_agent = SviAgent(
        model_name=MODEL_NAME,
        bot_class=GeminiAgent,
        bot_kwargs={},  # Add API key or other kwargs if needed
    )

    prompt_manager = PromptManager()

    working_folder = os.path.join(os.getcwd(), "dataset_multimodal", "test_LLMrerank")

    query_svi_folder = os.path.join(working_folder, "query_img")
    target_svi_folder = os.path.join(os.getcwd(), "dataset_multimodal", "query100@target1186", "target_pano_2022_1186places_extracted2")
    cosplace_json_path = os.path.join(working_folder, "benchmarkcosplace_results.json")
    prompt_list = [prompt_manager.get_prompt("survey_rerank", "rerank_1")]
    save_dir = os.path.join(working_folder, "LLMrerank_results")
    
    llm_result_path = svi_agent.svi_top100_batch_pipeline(prompt_list, query_svi_folder, target_svi_folder, cosplace_json_path, save_dir)

def test_preprocess():
    # Initialize components
    MODEL_NAME = "gemini-2.5-flash"
    # Pass the class, not an instance
    svi_agent = SviAgent(
        model_name=MODEL_NAME,
        bot_class=GeminiAgent,
        bot_kwargs={},  # Add API key or other kwargs if needed
    )    
    image_path = os.path.join(os.getcwd(), "test_agent_flood.png")
    save_path = os.path.join(os.getcwd(), "test_agent_flood_grid.jpg")
    processed_image = svi_agent.preprocess_image(image_path, max_size=1024, grid_num=4, grid_line=True, axis=False)
    processed_image.save(save_path)

if __name__ == "__main__":
    # run_llm_find_matches_top100()
    test_preprocess()