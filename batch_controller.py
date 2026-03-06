import os
import json
import time
import pickle
from typing import List, Dict, Any, Callable, Optional, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class BatchController:
    """
    A reusable batch processing controller that provides:
    1. Progress tracking with tqdm
    2. Periodic saving and resumption capabilities
    3. Common parameter management
    4. Minimal modification to existing pipeline functions
    5. Parallel processing with rate limiting
    """
    
    def __init__(self, 
                 save_dir: str,
                 checkpoint_interval: int = 10,
                 checkpoint_filename: str = "batch_checkpoint.json",
                 progress_filename: str = "batch_progress.json",
                 checkpoint_serializer: Optional[Tuple[Callable, Callable]] = None):
        """
        Initialize the batch controller.
        
        Args:
            save_dir: Directory to save checkpoints and progress
            checkpoint_interval: Save checkpoint every N items
            checkpoint_filename: Name of checkpoint file
            progress_filename: Name of progress tracking file
            checkpoint_serializer: Optional (dump, load) functions for checkpointing (e.g., (pickle.dump, pickle.load))
        """
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = os.path.join(save_dir, checkpoint_filename)
        self.progress_path = os.path.join(save_dir, progress_filename)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize state
        self.results = []
        self.processed_items = set()
        self.start_time = None
        self.total_items = 0
        
        # Serializer for checkpointing
        if checkpoint_serializer is not None:
            self._checkpoint_dump, self._checkpoint_load = checkpoint_serializer
            self._checkpoint_mode = 'b'  # binary mode
        else:
            self._checkpoint_dump, self._checkpoint_load = json.dump, json.load
            self._checkpoint_mode = ''   # text mode
        
    def run_batch(self, 
                  items: List[Any],
                  process_func: Callable,
                  common_params: Dict[str, Any] = None,
                  resume: bool = True,
                  desc: str = "Processing") -> List[Any]:
        """
        Run batch processing with progress tracking and resumption.
        
        Args:
            items: List of items to process
            process_func: Function to process each item (should accept item and **kwargs)
            common_params: Common parameters to pass to process_func
            resume: Whether to resume from checkpoint if available
            desc: Description for progress bar
            
        Returns:
            List of results from processing
        """
        if common_params is None:
            common_params = {}
            
        self.total_items = len(items)
        self.start_time = time.time()
        
        # Try to resume from checkpoint
        if resume:
            self._load_checkpoint()
        
        # Create progress bar
        pbar = tqdm(
            total=self.total_items,
            desc=desc,
            unit="items",
            initial=len(self.processed_items)
        )
        
        try:
            for i, item in enumerate(items):
                # Skip if already processed
                item_key = str(item)
                if item_key in self.processed_items:
                    continue
                    
                # Process item
                try:
                    result = process_func(item, **common_params)
                    self.results.append(result)
                    self.processed_items.add(item_key)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'Processed': len(self.processed_items),
                        'Remaining': self.total_items - len(self.processed_items),
                        'ETA': self._estimate_eta(len(self.processed_items))
                    })
                    
                    # Save checkpoint periodically
                    if (i + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                except Exception as e:
                    logger.error(f"Error processing item {item}: {str(e)}")
                    # Continue with next item
                    continue
                    
        finally:
            pbar.close()
            # Final save
            self._save_checkpoint()
            self._save_progress()
            
        return self.results
    
    def run_batch_parallel(self, 
                          items: List[Any],
                          process_func: Callable,
                          common_params: Dict[str, Any] = None,
                          resume: bool = True,
                          desc: str = "Processing",
                          max_workers: int = 4,
                          rate_limiter: Optional[Callable] = None,
                          retry: bool = False) -> List[Any]:
        """
        Run batch processing in parallel with progress tracking and resumption.
        
        Args:
            items: List of items to process
            process_func: Function to process each item (should accept item and **kwargs)
            common_params: Common parameters to pass to process_func
            resume: Whether to resume from checkpoint if available
            desc: Description for progress bar
            max_workers: Maximum number of parallel workers
            rate_limiter: Optional rate limiting function to call before each API request
            retry: If True, remove failed items from processed_items to retry them
            
        Returns:
            List of results from processing
        """
        if common_params is None:
            common_params = {}
            
        self.total_items = len(items)
        self.start_time = time.time()
        
        # Try to resume from checkpoint
        if resume:
            self._load_checkpoint()
            
        # If retry mode, filter out failed items from processed_items
        if retry and hasattr(self, 'failed_items'):
            # Remove failed items from processed_items so they can be retried
            self.processed_items = self.processed_items - self.failed_items
            logger.info(f"Retry mode: {len(self.failed_items)} failed items will be retried")
        
        # Store initial processed count for progress bar updates
        initial_processed_count = len(self.processed_items)
        
        # Filter out already processed items (compare using normalized string keys)
        items_to_process = [item for item in items if str(item) not in self.processed_items]
        
        if not items_to_process:
            logger.info("All items already processed")
            return self.results
        
        # Create progress bar with total items (including already processed)
        pbar = tqdm(
            total=self.total_items,  # Use total items for proper resume display
            desc=desc,
            unit="items",  # Changed from "item" to "items"
            initial=len(self.processed_items)  # Start from already processed items
        )
        
        # Thread-safe locks for updating shared state
        results_lock = threading.Lock()
        processed_lock = threading.Lock()
        
        # Track failed items for retry functionality
        if retry:
            self.failed_items = set()
        
        def process_with_rate_limit(item):
            """Wrapper function that applies rate limiting before processing"""
            if rate_limiter:
                rate_limiter()
            return process_func(item, **common_params)
        
        completed_this_run = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(process_with_rate_limit, item): item 
                    for item in items_to_process
                }
                
                # Process completed tasks
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        result = future.result()
                        
                        # Thread-safe updates
                        with results_lock:
                            self.results.append(result)
                        with processed_lock:
                            self.processed_items.add(str(item))
                            completed_this_run += 1
                        
                        # Always update progress bar for each completion
                        pbar.update(1)
                        processed_count = len(self.processed_items)
                        remaining = max(0, self.total_items - processed_count)
                        pbar.set_postfix({
                            'Processed': processed_count,
                            'Remaining': remaining,
                            'ETA': self._estimate_eta(processed_count)
                        })
                        
                        # Save checkpoint periodically
                        if processed_count % self.checkpoint_interval == 0:
                            self._save_checkpoint()
                            logger.debug(f"Checkpoint saved at {processed_count} items")
                            
                    except Exception as e:
                        logger.error(f"Error processing item {item}: {str(e)}")
                        
                        # In retry mode, track failed items
                        if retry:
                            with processed_lock:
                                self.failed_items.add(item)
                        
                        # Continue with next item
                        continue
                        
        finally:
            pbar.close()
            # Final save
            self._save_checkpoint()
            self._save_progress()
            
        return self.results
    
    def _to_serializable(self, obj):
        """Recursively convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {self._to_serializable(k): self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self._to_serializable(i) for i in obj]
        elif hasattr(obj, 'as_posix'):  # Path objects
            return str(obj)
        elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):  # Numpy arrays
            return f"numpy_array(shape={obj.shape}, dtype={obj.dtype})"
        else:
            return obj
    
    def _save_checkpoint(self):
        """Save current state to checkpoint file."""
        # For pickle serialization, save results directly without conversion
        if self._checkpoint_mode == 'b':
            checkpoint_data = {
                'results': self.results,
                'processed_items': [str(item) for item in self.processed_items],
                'timestamp': time.time()
            }
            
            # Add failed items if tracking them
            if hasattr(self, 'failed_items'):
                checkpoint_data['failed_items'] = [str(item) for item in self.failed_items]
        else:
            # For JSON serialization, use _to_serializable
            checkpoint_data = {
                'results': self._to_serializable(self.results),
                'processed_items': [str(item) for item in self.processed_items],
                'timestamp': time.time()
            }
            
            # Add failed items if tracking them
            if hasattr(self, 'failed_items'):
                checkpoint_data['failed_items'] = [str(item) for item in self.failed_items]
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

            # Prepare atomic write paths
            tmp_path = self.checkpoint_path + ".tmp"
            bak_path = self.checkpoint_path + ".bak"

            # Rotate existing checkpoint to .bak
            if os.path.exists(self.checkpoint_path):
                try:
                    # Remove stale .bak if present to avoid growing files
                    if os.path.exists(bak_path):
                        try:
                            os.remove(bak_path)
                        except Exception:
                            pass
                    os.replace(self.checkpoint_path, bak_path)
                except Exception:
                    # Non-fatal; continue to attempt writing new checkpoint
                    pass

            # Write to temporary file first
            mode = 'w' + self._checkpoint_mode
            with open(tmp_path, mode) as f:
                if self._checkpoint_mode == '':
                    self._checkpoint_dump(checkpoint_data, f, indent=2)
                    f.flush()
                else:
                    self._checkpoint_dump(checkpoint_data, f)
                    try:
                        f.flush()
                    except Exception:
                        pass
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # fsync may not be available on some platforms; ignore
                    pass

            # Atomically replace
            os.replace(tmp_path, self.checkpoint_path)

            logger.debug(f"Checkpoint saved successfully with {len(self.results)} results and {len(self.processed_items)} processed items")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def _load_checkpoint(self):
        """Load state from checkpoint file."""
        if not os.path.exists(self.checkpoint_path):
            return
        
        try:
            mode = 'r' + self._checkpoint_mode
            with open(self.checkpoint_path, mode) as f:
                checkpoint_data = self._checkpoint_load(f)
            
            # Validate checkpoint data
            if not isinstance(checkpoint_data, dict):
                logger.error("Invalid checkpoint format: not a dictionary")
                return
            
            # Load results directly (no conversion needed for pickle)
            self.results = checkpoint_data.get('results', [])
            self.processed_items = set(checkpoint_data.get('processed_items', []))
            
            # Load failed items if they exist
            if 'failed_items' in checkpoint_data:
                self.failed_items = set(checkpoint_data.get('failed_items', []))
            
            logger.info(f"Resumed from checkpoint: {len(self.processed_items)} items already processed, {len(self.results)} results loaded")
            if hasattr(self, 'failed_items'):
                logger.info(f"Failed items from checkpoint: {len(self.failed_items)}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            # Attempt to recover from backup
            bak_path = self.checkpoint_path + ".bak"
            if os.path.exists(bak_path):
                try:
                    mode = 'r' + self._checkpoint_mode
                    with open(bak_path, mode) as f:
                        checkpoint_data = self._checkpoint_load(f)
                    if isinstance(checkpoint_data, dict):
                        self.results = checkpoint_data.get('results', [])
                        self.processed_items = set(checkpoint_data.get('processed_items', []))
                        if 'failed_items' in checkpoint_data:
                            self.failed_items = set(checkpoint_data.get('failed_items', []))
                        logger.warning("Recovered from backup checkpoint (.bak)")
                        return
                except Exception as e_bak:
                    logger.error(f"Failed to load backup checkpoint: {str(e_bak)}")

            # If checkpoint is corrupted and no backup, start fresh
            logger.info("Starting fresh due to checkpoint loading failure")
            self.results = []
            self.processed_items = set()
    
    def _save_progress(self):
        """Save final progress summary."""
        progress_data = {
            'total_items': self.total_items,
            'processed_items': len(self.processed_items),
            'success_rate': len(self.processed_items) / self.total_items if self.total_items > 0 else 0,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'timestamp': time.time()
        }
        
        try:
            with open(self.progress_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")
    
    def _estimate_eta(self, processed_count: int) -> str:
        """Estimate remaining time."""
        if processed_count == 0 or self.start_time is None:
            return "Unknown"
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return "Unknown"
            
        avg_time_per_item = elapsed_time / processed_count
        remaining_items = self.total_items - processed_count
        eta_seconds = avg_time_per_item * remaining_items
        
        # Convert to human readable format
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.0f}m"
        else:
            return f"{eta_seconds/3600:.1f}h"
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        return {
            'total_items': self.total_items,
            'processed_items': len(self.processed_items),
            'remaining_items': self.total_items - len(self.processed_items),
            'success_rate': len(self.processed_items) / self.total_items if self.total_items > 0 else 0,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the current checkpoint."""
        checkpoint_exists = os.path.exists(self.checkpoint_path)
        checkpoint_size = os.path.getsize(self.checkpoint_path) if checkpoint_exists else 0
        
        return {
            'checkpoint_exists': checkpoint_exists,
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_size_bytes': checkpoint_size,
            'checkpoint_size_mb': checkpoint_size / (1024 * 1024) if checkpoint_exists else 0
        } 