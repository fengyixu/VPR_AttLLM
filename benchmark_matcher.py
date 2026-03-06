import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if _os.name == "nt":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
import logging
import pickle
import faiss
import numpy as np
import pandas as pd
import json
import gc
import torch
from tqdm import tqdm
from project_utils import safe_json_load, get_filtered_features_and_index

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)


class BenchmarkMatcher:
    """
    End-to-end VPR pipeline for CosPlace (baseline) and AttentionCosPlace (LLM-guided).

    Responsibilities:
    - Load the CosPlace model once and reuse it across multiple runs.
    - Build / load a FAISS-indexed reference feature database.
    - Extract query features, optionally modulated by per-image LLM attention maps.
    - Retrieve the top-k database candidates for every query.
    - Optionally apply Average Query Expansion (AQE) for a second retrieval pass.
    """

    def __init__(self, vpr_model: str = "cosplace"):
        """
        Args:
            vpr_model: Must be "cosplace".
        """
        logger.info("Initializing BenchmarkMatcher")
        if vpr_model == "cosplace":
            from benchmark_models import Cosplace
            self.model_handler = Cosplace()
        else:
            raise ValueError(
                f"Unsupported VPR model: '{vpr_model}'. Only 'cosplace' is supported in this release."
            )
        self.transform = None

        from benchmark_feature import BenchmarkFeature
        self.feature_generator = BenchmarkFeature(vpr_model)

        self._distributed_index_cache = {}
        self._gpu_resources = None
        self._max_cached_indices = 5

    # ------------------------------------------------------------------
    # Average Query Expansion
    # ------------------------------------------------------------------

    def average_query_expansion(self, query_features_dict, reference_features, record_dict, k=5, alpha=0.8):
        """
        Lightweight Average Query Expansion (AQE).

        Re-weights each query descriptor as:
            q_expanded = α * q + (1 - α) * mean(top-k reference descriptors)

        Args:
            query_features_dict: {filename: descriptor}.
            reference_features: List of reference feature records.
            record_dict: First-pass retrieval results.
            k: Number of top matches used for expansion.
            alpha: Weight of the original query (default 0.8).

        Returns:
            Updated {filename: descriptor} mapping.
        """
        if isinstance(reference_features, dict) and reference_features.get('type') == 'distributed':
            raise ValueError("AQE is not supported for distributed (large-scale) reference features.")

        reference_lookup = {}
        for entry in reference_features:
            features = entry.get('features') if isinstance(entry, dict) else entry
            vector = np.asarray(features, dtype=np.float32).reshape(-1)
            key = os.path.basename(entry['path']) if isinstance(entry, dict) and 'path' in entry else None
            if key is None:
                raise ValueError(f"No path key found in reference entry: {entry}")
            reference_lookup[key] = vector

        expanded = {}
        for query_name, query_features in query_features_dict.items():
            if query_features is None:
                expanded[query_name] = None
                continue
            query_vector = np.asarray(query_features, dtype=np.float32).reshape(-1)
            top_targets = record_dict.get(query_name, {}).get("target_path", [])
            if not top_targets:
                raise ValueError(f"No top-k targets for query {query_name}")
            top_descriptors = []
            for target in top_targets[:max(1, k)]:
                ref_vec = reference_lookup.get(target)
                if ref_vec is None:
                    raise ValueError(f"Missing reference features for target {target}")
                top_descriptors.append(np.asarray(ref_vec, dtype=np.float32).reshape(-1))
            stacked = np.vstack(top_descriptors)
            q_expanded = alpha * query_vector + (1 - alpha) * stacked.sum(axis=0) / len(top_descriptors)
            expanded[query_name] = q_expanded.astype(np.float32)
        return expanded

    # ------------------------------------------------------------------
    # Internal batch-query helpers
    # ------------------------------------------------------------------

    def cleanup_distributed_cache(self):
        self._distributed_index_cache.clear()
        if self._gpu_resources is not None:
            try:
                del self._gpu_resources
                self._gpu_resources = None
            except Exception:
                pass
        gc.collect()

    def _process_distributed_batch_queries(self, query_features_dict, distributed_index,
                                            top_k, rerank, initial_result_dict,
                                            save_dir, record_dict_path):
        total_queries = len(query_features_dict)
        logger.info(f"Distributed batch: {total_queries} queries across {len(distributed_index['chunks'])} chunks")

        all_results = []
        for chunk_info in tqdm(distributed_index['chunks'], desc="Processing distributed chunks"):
            chunk_file = chunk_info['chunk_file']
            if not os.path.exists(chunk_file):
                continue
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_features = pickle.load(f)
                if not chunk_features:
                    continue
                features_array = np.array([e['features'] for e in chunk_features]).astype('float32')
                if features_array.size == 0:
                    continue
                use_gpu = torch.cuda.is_available()
                chunk_index = faiss.IndexFlatL2(features_array.shape[1])
                if use_gpu:
                    try:
                        if self._gpu_resources is None:
                            self._gpu_resources = faiss.StandardGpuResources()
                        chunk_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, chunk_index)
                    except Exception:
                        pass
                chunk_index.add(features_array)
                for query_filename, query_features in query_features_dict.items():
                    q_arr = query_features.reshape(1, -1).astype('float32')
                    distances, indices = chunk_index.search(q_arr, top_k)
                    for i in range(min(top_k, len(indices[0]))):
                        idx = indices[0][i]
                        all_results.append({
                            'query_path': query_filename,
                            'similarity_score': -float(distances[0][i]),
                            'target_path': os.path.basename(chunk_features[idx]['path']),
                        })
                del chunk_index, features_array, chunk_features
                gc.collect()
            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk_file}: {e}")

        record_dict = {}
        for result in all_results:
            qp = result['query_path']
            if qp not in record_dict:
                record_dict[qp] = {"target_path": [], "similarity_score": []}
            record_dict[qp]["target_path"].append(result['target_path'])
            record_dict[qp]["similarity_score"].append(result['similarity_score'])
        for qp in record_dict:
            pairs = sorted(zip(record_dict[qp]["target_path"], record_dict[qp]["similarity_score"]),
                           key=lambda x: x[1], reverse=True)[:top_k]
            record_dict[qp]["target_path"] = [p[0] for p in pairs]
            record_dict[qp]["similarity_score"] = [p[1] for p in pairs]

        pd.DataFrame(record_dict).to_json(record_dict_path, index=False)
        logger.info(f"Saved distributed results to {record_dict_path}")
        self.cleanup_distributed_cache()
        return record_dict_path, record_dict

    def _process_traditional_batch_queries(self, query_features_dict, reference_features,
                                            faiss_index, top_k, rerank, initial_result_dict,
                                            save_dir, record_dict_path):
        total_queries = len(query_features_dict)
        logger.info(f"Traditional batch: {total_queries} queries")
        all_results = []

        query_filenames = list(query_features_dict.keys())
        query_features_list = [query_features_dict[fn] for fn in query_filenames]

        try:
            normalized = [np.asarray(f, dtype=np.float32).flatten() for f in query_features_list if f is not None]
            if normalized:
                query_arr = np.array(normalized).astype('float32')
                distances, indices = faiss_index.search(query_arr, top_k)
                for qi, query_filename in enumerate(query_filenames):
                    for i in range(min(top_k, len(indices[qi]))):
                        idx = indices[qi][i]
                        all_results.append({
                            'query_path': query_filename,
                            'similarity_score': -float(distances[qi][i]),
                            'target_path': os.path.basename(reference_features[idx]['path']),
                        })
        except Exception as e:
            logger.warning(f"Batch FAISS search failed ({e}); falling back to individual queries")
            for query_filename, query_features in tqdm(query_features_dict.items(), desc="Querying (individual)"):
                try:
                    q_arr = query_features.reshape(1, -1).astype('float32')
                    distances, indices = faiss_index.search(q_arr, top_k)
                    for i in range(min(top_k, len(indices[0]))):
                        idx = indices[0][i]
                        all_results.append({
                            'query_path': query_filename,
                            'similarity_score': -float(distances[0][i]),
                            'target_path': os.path.basename(reference_features[idx]['path']),
                        })
                except Exception as e2:
                    logger.warning(f"Failed query {query_filename}: {e2}")

        record_dict = {}
        for result in all_results:
            qp = result['query_path']
            if qp not in record_dict:
                record_dict[qp] = {"target_path": [], "similarity_score": []}
            record_dict[qp]["target_path"].append(result['target_path'])
            record_dict[qp]["similarity_score"].append(result['similarity_score'])

        pd.DataFrame(record_dict).to_json(record_dict_path, index=False)
        logger.info(f"Saved traditional results to {record_dict_path}")
        return record_dict_path, record_dict

    # ------------------------------------------------------------------
    # Public pipeline entry point
    # ------------------------------------------------------------------

    def benchmark_pipeline_batch(self, query_svi_folder, target_svi_folder, save_dir,
                                   database_dir, top_k=5, max_workers=4, batch_size=64,
                                   llm_att=False, llm_json_folder=None, rerank=False,
                                   initial_result_json=None, att_ratio=0.1,
                                   interpolate=False, qe=False):
        """
        Full VPR retrieval pipeline.

        Steps:
        1. Build / load reference database (FAISS-indexed CosPlace descriptors).
        2. Extract query descriptors (with optional LLM attention via AttentionCosPlace).
        3. Retrieve top-k candidates per query using L2 nearest-neighbour search.
        4. Optionally apply Average Query Expansion for a second retrieval pass.

        Args:
            query_svi_folder: Folder with query images.
            target_svi_folder: Folder with database images.
            save_dir: Directory for result JSON files.
            database_dir: Directory for caching reference features.
            top_k: Number of top candidates to retrieve per query.
            max_workers: Workers for parallel shard generation (always 1 recommended for GPU).
            batch_size: Images per forward pass.
            llm_att: Use LLM attention (requires llm_json_folder).
            llm_json_folder: Folder with per-image JSON attention files.
            rerank: (reserved) Currently unused.
            initial_result_json: (reserved) Currently unused.
            att_ratio: Attention blend factor α ∈ [0, 1].
            interpolate: Bilinear interpolation of grid attention map.
            qe: Apply Average Query Expansion after first retrieval pass.

        Returns:
            Path to the result JSON file.
        """
        model_name = self.model_handler.__class__.__name__.lower()
        record_dict_filename = f"{model_name}_results.json"
        record_dict_path = os.path.join(save_dir, record_dict_filename)

        initial_result_dict = None
        if rerank and initial_result_json is not None:
            initial_result_dict = safe_json_load(initial_result_json)
            if initial_result_dict is None:
                logger.warning("Failed to load initial results; disabling rerank")
                rerank = False

        if os.path.exists(record_dict_path):
            logger.info(f"Result already exists: {record_dict_path}")
            return record_dict_path

        self.model_handler.load_model()
        self.transform = self.model_handler.setup_image_transform()

        reference_features, faiss_index = self.feature_generator.load_or_generate_features(
            target_svi_folder, database_dir, max_workers=max_workers, batch_size=batch_size)
        logger.info(f"Loaded {len(reference_features)} reference features")

        query_files = [f for f in os.listdir(query_svi_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        query_paths = [os.path.join(query_svi_folder, f) for f in query_files]

        query_features_dict = self.feature_generator.generate_query_features_batch(
            query_paths, batch_size=batch_size,
            llm_att=llm_att, llm_json_folder=llm_json_folder,
            att_ratio=att_ratio, interpolate=interpolate,
        )

        is_distributed = isinstance(faiss_index, dict) and faiss_index.get('type') == 'distributed'

        if is_distributed:
            record_dict_path, record_dict = self._process_distributed_batch_queries(
                query_features_dict, faiss_index,
                top_k, rerank, initial_result_dict, save_dir, record_dict_path)
        else:
            record_dict_path, record_dict = self._process_traditional_batch_queries(
                query_features_dict, reference_features, faiss_index,
                top_k, rerank, initial_result_dict, save_dir, record_dict_path)

        if qe and not is_distributed:
            logger.info("Applying Average Query Expansion (AQE)...")
            try:
                expanded = self.average_query_expansion(query_features_dict, reference_features, record_dict)
                if expanded:
                    record_dict_path, _ = self._process_traditional_batch_queries(
                        expanded, reference_features, faiss_index,
                        top_k, rerank, initial_result_dict, save_dir, record_dict_path)
            except Exception as e:
                logger.warning(f"AQE failed; returning first-pass results: {e}")

        return record_dict_path
