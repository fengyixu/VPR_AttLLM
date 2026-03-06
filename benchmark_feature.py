import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if _os.name == "nt":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
import logging
import time
import pickle
import faiss
import numpy as np
import json
import gc
import torch
import psutil
from typing import List
from tqdm import tqdm
from batch_controller import BatchController
from project_utils import safe_json_load

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)


class BenchmarkFeature:
    """
    Feature generation and database management for CosPlace VPR.

    Handles batched feature extraction, shard-based persistence with
    resume support, streaming consolidation, and FAISS index construction.
    """

    def __init__(self, vpr_model: str):
        """
        Args:
            vpr_model: Must be "cosplace".
        """
        logger.info("Initializing BenchmarkFeature")
        if vpr_model == "cosplace":
            from benchmark_models import Cosplace
            self.model_handler = Cosplace()
        else:
            raise ValueError(
                f"Unsupported VPR model: '{vpr_model}'. Only 'cosplace' is supported in this release."
            )
        self.transform = None

    # ------------------------------------------------------------------
    # Memory / shard helpers
    # ------------------------------------------------------------------

    def _get_adaptive_chunk_size(self, total_shards, available_memory_gb=None):
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        estimated_memory_per_feature = 2 * 1024
        features_per_shard = 64
        safe_memory_bytes = available_memory_gb * 1024 ** 3 * 0.4
        max_features_in_memory = safe_memory_bytes // estimated_memory_per_feature
        optimal_chunk_size = max(1, max_features_in_memory // features_per_shard)
        return int(min(optimal_chunk_size, total_shards))

    def _verify_completion_before_consolidation(self, shard_dir, batch_indices):
        logger.info("Verifying completion before consolidation...")
        missing_shards, corrupted_shards = [], []
        total_found = 0
        for batch_index in batch_indices:
            shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
            if not os.path.exists(shard_path):
                missing_shards.append(batch_index)
                continue
            try:
                with open(shard_path, 'rb') as f:
                    records = pickle.load(f)
                if isinstance(records, list):
                    total_found += 1
                else:
                    corrupted_shards.append(batch_index)
            except Exception as e:
                corrupted_shards.append(batch_index)
                logger.warning(f"Corrupted shard {shard_path}: {e}")
        total_expected = len(batch_indices)
        completion_rate = total_found / total_expected if total_expected > 0 else 0
        status = {
            'total_expected': total_expected,
            'total_found': total_found,
            'missing_shards': missing_shards,
            'corrupted_shards': corrupted_shards,
            'completion_rate': completion_rate,
        }
        logger.info(f"Completion: {total_found}/{total_expected} ({completion_rate:.1%})")
        return status

    def _process_missing_batches(self, missing_batches, batched_paths, shard_dir):
        logger.info(f"Auto-processing {len(missing_batches)} missing/corrupted batches...")
        for batch_index in missing_batches:
            try:
                batch_paths = batched_paths[batch_index]
                shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
                features_array = self.model_handler.extract_features_batch(batch_paths, self.transform)
                if features_array is None:
                    with open(shard_path, 'wb') as f:
                        pickle.dump([], f)
                else:
                    shard_records = [{'features': features_array[idx], 'path': batch_paths[idx]}
                                     for idx in range(len(batch_paths))]
                    tmp_path = shard_path + '.tmp'
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(shard_records, f)
                        try:
                            f.flush(); os.fsync(f.fileno())
                        except Exception:
                            pass
                    os.replace(tmp_path, shard_path)
            except Exception as e:
                logger.error(f"Failed to process batch {batch_index}: {e}")

    def _consolidate_shards_distributed(self, shard_dir, save_path, batch_indices, batched_paths):
        logger.info("Analyzing dataset size for consolidation strategy...")
        status = self._verify_completion_before_consolidation(shard_dir, batch_indices)
        if status['completion_rate'] < 1.0:
            missing = status['missing_shards'] + status['corrupted_shards']
            if missing:
                self._process_missing_batches(missing, batched_paths, shard_dir)
        total_size_gb = self._calculate_shards_total_size(shard_dir, batch_indices)
        size_threshold_gb = 6.0
        if total_size_gb > size_threshold_gb:
            return self._consolidate_large_dataset_distributed(shard_dir, save_path, batch_indices, total_size_gb)
        else:
            return self._consolidate_small_dataset_streaming(shard_dir, save_path, batch_indices)

    def _calculate_shards_total_size(self, shard_dir, batch_indices):
        total_size_bytes = 0
        for batch_index in batch_indices:
            shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
            if os.path.exists(shard_path):
                try:
                    total_size_bytes += os.path.getsize(shard_path)
                except OSError:
                    continue
        return total_size_bytes / (1024 ** 3)

    def _consolidate_large_dataset_distributed(self, shard_dir, save_path, batch_indices, total_size_gb):
        distributed_dir = os.path.join(os.path.dirname(save_path), 'distributed_features')
        os.makedirs(distributed_dir, exist_ok=True)
        target_chunk_size_gb = 5.0
        estimated_chunks = max(1, int(total_size_gb / target_chunk_size_gb))
        shards_per_chunk = max(1, len(batch_indices) // estimated_chunks)
        chunk_info = []
        total_features = 0
        for chunk_idx in tqdm(range(estimated_chunks), desc="Consolidating chunks"):
            chunk_start = chunk_idx * shards_per_chunk
            chunk_end = min(chunk_start + shards_per_chunk, len(batch_indices))
            chunk_batch_indices = batch_indices[chunk_start:chunk_end]
            chunk_file = os.path.join(distributed_dir, f"features_chunk_{chunk_idx:04d}.pkl")
            chunk_features = self._consolidate_chunk(shard_dir, chunk_file, chunk_batch_indices)
            if chunk_features > 0:
                chunk_info.append({
                    'chunk_file': chunk_file,
                    'feature_count': chunk_features,
                    'chunk_size_mb': os.path.getsize(chunk_file) / (1024 ** 2) if os.path.exists(chunk_file) else 0,
                    'batch_range': (chunk_start, chunk_end),
                })
                total_features += chunk_features
        distributed_index = {
            'total_features': total_features,
            'total_chunks': len(chunk_info),
            'total_size_gb': total_size_gb,
            'target_chunk_size_gb': target_chunk_size_gb,
            'chunks': chunk_info,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        index_file = os.path.join(distributed_dir, 'distributed_index.json')
        with open(index_file, 'w') as f:
            json.dump(distributed_index, f, indent=2)
        reference_info = {
            'type': 'distributed',
            'index_file': index_file,
            'total_features': total_features,
            'chunk_count': len(chunk_info),
            'total_size_gb': total_size_gb,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(reference_info, f)
        logger.info(f"Distributed consolidation complete: {total_features} features in {len(chunk_info)} chunks")
        return total_features

    def _consolidate_chunk(self, shard_dir, chunk_file, batch_indices):
        chunk_features = []
        for batch_index in batch_indices:
            shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
            if not os.path.exists(shard_path):
                continue
            try:
                with open(shard_path, 'rb') as f:
                    records = pickle.load(f)
                if records:
                    chunk_features.extend(records)
            except Exception as e:
                logger.warning(f"Failed to read shard {shard_path}: {e}")
        if chunk_features:
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_features, f)
            return len(chunk_features)
        return 0

    def _consolidate_small_dataset_streaming(self, shard_dir, save_path, batch_indices):
        logger.info("Small dataset: streaming consolidation...")
        chunk_size = self._get_adaptive_chunk_size(len(batch_indices))
        total_features = 0
        for chunk_start in tqdm(range(0, len(batch_indices), chunk_size), desc="Streaming consolidation"):
            chunk_end = min(chunk_start + chunk_size, len(batch_indices))
            chunk_features = []
            for batch_index in batch_indices[chunk_start:chunk_end]:
                shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
                if not os.path.exists(shard_path):
                    continue
                try:
                    with open(shard_path, 'rb') as f:
                        records = pickle.load(f)
                    if records:
                        chunk_features.extend(records)
                except Exception as e:
                    logger.warning(f"Failed to read shard {shard_path}: {e}")
            if chunk_features:
                if total_features == 0:
                    with open(save_path, 'wb') as f:
                        pickle.dump(chunk_features, f)
                else:
                    with open(save_path, 'rb') as f:
                        existing = pickle.load(f)
                    existing.extend(chunk_features)
                    with open(save_path, 'wb') as f:
                        pickle.dump(existing, f)
                    del existing
                total_features += len(chunk_features)
                del chunk_features
                gc.collect()
        backup_marker = os.path.join(shard_dir, '.consolidation_complete')
        with open(backup_marker, 'w') as f:
            f.write(f"Consolidated {total_features} features on {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._verify_consolidated_file(save_path, total_features)
        return total_features

    def _verify_consolidated_file(self, save_path, expected_features):
        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                raise ValueError("Consolidated file is not a list")
            actual = len(data)
            if actual != expected_features:
                logger.warning(f"Feature count mismatch: expected {expected_features}, got {actual}")
            else:
                logger.info(f"Consolidated file verified: {actual} features")
        except Exception as e:
            logger.error(f"Failed to verify consolidated file: {e}")
            raise

    # ------------------------------------------------------------------
    # Main generation pipeline
    # ------------------------------------------------------------------

    def _generate_features_batch(self, target_svi_list, database_dir, checkpoint_interval=20,
                                  resume=True, desc="Generating reference features",
                                  max_workers=4, batch_size=64, transform=None):
        batch_controller = BatchController(
            save_dir=database_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint_filename="batch_checkpoint.pkl",
            checkpoint_serializer=(pickle.dump, pickle.load)
        )
        batched_paths = [target_svi_list[i:i + batch_size]
                         for i in range(0, len(target_svi_list), batch_size)]
        batch_indices = list(range(len(batched_paths)))
        shard_dir = os.path.join(database_dir, 'ref_shards')
        os.makedirs(shard_dir, exist_ok=True)

        def process_batch(batch_index):
            batch_paths = batched_paths[batch_index]
            shard_path = os.path.join(shard_dir, f"ref_batch_{batch_index:06d}.pkl")
            if os.path.exists(shard_path):
                try:
                    with open(shard_path, 'rb') as f:
                        test_data = pickle.load(f)
                    if isinstance(test_data, list):
                        return {"shard": os.path.basename(shard_path), "count": len(test_data)}
                except Exception as e:
                    logger.warning(f"Corrupted shard {shard_path}, reprocessing: {e}")
                    try:
                        os.remove(shard_path)
                    except Exception:
                        pass
            features_array = self.model_handler.extract_features_batch(batch_paths, self.transform)
            if features_array is None:
                with open(shard_path, 'wb') as f:
                    pickle.dump([], f)
                return {"shard": os.path.basename(shard_path), "count": 0}
            shard_records = [{'features': features_array[idx], 'path': batch_paths[idx]}
                              for idx in range(len(batch_paths))]
            tmp_path = shard_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(shard_records, f)
                try:
                    f.flush(); os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, shard_path)
            return {"shard": os.path.basename(shard_path), "count": len(shard_records)}

        adaptive_workers = min(max_workers, 2) if max_workers > 1 else 1
        batch_controller.run_batch_parallel(
            items=batch_indices,
            process_func=process_batch,
            common_params={},
            resume=resume,
            desc=desc,
            max_workers=adaptive_workers,
        )

        logger.info("Consolidating shards into reference_features.pkl...")
        save_path = os.path.join(database_dir, 'reference_features.pkl')
        try:
            total_features = self._consolidate_shards_distributed(shard_dir, save_path, batch_indices, batched_paths)
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            raise

        logger.info(f"Feature generation completed: {total_features} features")
        try:
            with open(save_path, 'rb') as f:
                reference_info = pickle.load(f)
            if isinstance(reference_info, dict) and reference_info.get('type') == 'distributed':
                return reference_info
            return reference_info
        except MemoryError:
            logger.warning("Cannot load full reference_features into memory.")
            return []

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _build_faiss_index(self, reference_features):
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass
        if isinstance(reference_features, dict) and reference_features.get('type') == 'distributed':
            return self._build_distributed_faiss_index(reference_features)
        if not reference_features:
            logger.warning("No reference features; creating empty FAISS index (dim=512)")
            return faiss.IndexFlatL2(512)
        features_array = np.array([e['features'] for e in reference_features]).astype('float32')
        if features_array.size == 0:
            return faiss.IndexFlatL2(512)
        dimension = features_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(features_array)
        return index

    def _build_distributed_faiss_index(self, distributed_info):
        with open(distributed_info['index_file'], 'r') as f:
            index_data = json.load(f)
        return {
            'type': 'distributed',
            'index_file': distributed_info['index_file'],
            'total_features': distributed_info['total_features'],
            'chunk_count': distributed_info['chunk_count'],
            'chunks': index_data['chunks'],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_or_generate_features(self, target_svi_folder, database_dir, max_workers=4, batch_size=64):
        """
        Load existing reference features from disk or generate them from scratch.

        Args:
            target_svi_folder: Folder containing database images.
            database_dir: Directory to cache feature files.
            max_workers: Workers for parallel shard writing.
            batch_size: Images per forward pass.

        Returns:
            (reference_features, faiss_index)
        """
        os.makedirs(database_dir, exist_ok=True)
        features_path = os.path.join(database_dir, 'reference_features.pkl')

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        list_file = os.path.join(target_svi_folder, 'database_images_paths.txt')
        target_svi_list: list = []
        if os.path.exists(list_file):
            try:
                with open(list_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        rel_path = line.strip()
                        if not rel_path or not rel_path.lower().endswith(valid_exts):
                            continue
                        target_svi_list.append(os.path.join(target_svi_folder, rel_path))
            except Exception as e:
                logger.warning(f"Failed to read {list_file}: {e}. Falling back to folder scan.")
        if not target_svi_list:
            target_svi_list = sorted(
                os.path.join(target_svi_folder, f)
                for f in os.listdir(target_svi_folder)
                if f.lower().endswith(valid_exts)
            )
            logger.info(f"Found {len(target_svi_list)} images in database folder")

        if os.path.exists(features_path):
            try:
                logger.info(f"Loading pre-computed features from {features_path}")
                with open(features_path, 'rb') as f:
                    reference_features = pickle.load(f)
                if isinstance(reference_features, dict) and reference_features.get('type') == 'distributed':
                    return reference_features, self._build_faiss_index(reference_features)
                if not isinstance(reference_features, list):
                    raise ValueError("Invalid reference_features format")
                if len(reference_features) == len(target_svi_list):
                    logger.info(f"Loaded database with {len(target_svi_list)} features")
                else:
                    logger.info(f"Feature count mismatch; regenerating database...")
                    reference_features = self._generate_features_batch(
                        target_svi_list, database_dir=database_dir,
                        max_workers=max_workers, batch_size=batch_size)
            except Exception as e:
                logger.warning(f"Failed to load features: {e}. Regenerating...")
                reference_features = self._generate_features_batch(
                    target_svi_list, database_dir=database_dir,
                    max_workers=max_workers, batch_size=batch_size)
        else:
            logger.info(f"Generating feature database ({len(target_svi_list)} images)...")
            reference_features = self._generate_features_batch(
                target_svi_list, database_dir=database_dir,
                max_workers=max_workers, batch_size=batch_size)

        if isinstance(reference_features, dict) and reference_features.get('type') == 'distributed':
            faiss_index = self._build_faiss_index(reference_features)
        else:
            reference_features = [f for f in reference_features if f is not None]
            faiss_index = self._build_faiss_index(reference_features)

        return reference_features, faiss_index

    def generate_query_features_batch(self, query_paths, batch_size=16, llm_att=False,
                                       llm_json_folder=None, att_ratio=0.1, interpolate=True):
        """
        Extract query features (with or without LLM attention).

        Args:
            query_paths: List of absolute query image paths.
            batch_size: Images per batch (only for standard extraction).
            llm_att: Enable LLM-guided attention.
            llm_json_folder: Folder containing per-image JSON attention files.
            att_ratio: Attention blend factor ∈ [0, 1].
            interpolate: Bilinear interpolation of grid attention map.

        Returns:
            {filename: feature_vector} dict.
        """
        logger.info(f"Generating query features for {len(query_paths)} queries (batch_size={batch_size})")
        query_features_dict = {}

        if llm_att and llm_json_folder is not None and att_ratio > 0:
            for path in tqdm(query_paths, desc="Extracting query features with attention"):
                filename = os.path.basename(path)
                try:
                    query_basename = os.path.splitext(filename)[0] + ".json"
                    llm_data = safe_json_load(os.path.join(llm_json_folder, query_basename))
                    llm_dict = None if llm_data is None else llm_data.get('result')
                    if llm_dict is None:
                        features = self.model_handler.extract_features(path, self.transform)
                    else:
                        features = self._extract_single_attention_features(path, llm_dict, att_ratio, interpolate)
                    if features is not None:
                        query_features_dict[filename] = features
                except Exception as e:
                    logger.warning(f"Failed for {filename}: {e}")
        else:
            # CosPlace uses variable-sized inputs; extract individually
            logger.info("Extracting query features individually (variable-size transform)")
            for path in tqdm(query_paths, desc="Extracting query features"):
                filename = os.path.basename(path)
                try:
                    features = self.model_handler.extract_features(path, self.transform)
                    if features is not None:
                        query_features_dict[filename] = features
                except Exception as e:
                    logger.warning(f"Failed for {filename}: {e}")

        logger.info(f"Extracted features for {len(query_features_dict)}/{len(query_paths)} queries")
        return query_features_dict

    def _extract_single_attention_features(self, query_path, llm_dict, att_ratio, interpolate):
        """Extract attention-weighted descriptor for a single query image."""
        try:
            from att_models import AttentionCosPlace
            agent_att = AttentionCosPlace(model_handler=self.model_handler, transform=self.transform)
            query_features = agent_att.extract_features_with_attention(
                query_path, llm_dict, att_ratio=att_ratio, interpolate=interpolate
            )
            if query_features is not None and hasattr(query_features, 'detach'):
                query_features = query_features.detach().cpu().numpy()
            return query_features
        except Exception as e:
            logger.warning(f"Attention extraction failed for {query_path}: {e}. Falling back.")
            return self.model_handler.extract_features(query_path, self.transform)
