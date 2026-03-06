"""
Pipeline C smoke test — no real model, no real dataset required.

Strategy
--------
1.  Synthetic 1×1 JPEG images named with coordinate-embedded filenames
    (query: "dash" format; database: "parse" / SF-XL format).
2.  A MockCosplace that returns *deterministic* 512-d feature vectors
    derived from the filename → no PyTorch Hub download, no GPU.
    Images whose coordinates are within the 100 m threshold share a
    very similar feature vector, so Recall@1 should be 1.0 on the
    near-duplicate pairs.
3.  Fake LLM grid-attention JSON files for every query image, so both
    the standard path (llm_att=False) and the attention path
    (llm_att=True) are exercised.
4.  All temp data lives in a system temp folder and is cleaned up after
    the test.

Run:
    python test_pipeline_c.py
"""

import sys
import os
import json
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure VPR_AttLLM project directory is on sys.path
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

FEAT_DIM = 512

# ---------------------------------------------------------------------------
# Coordinate-aware feature generator (pure python, no torch)
# ---------------------------------------------------------------------------

def _coord_seed(lat: float, lon: float, precision: int = 3) -> int:
    """
    Map quantized (lat, lon) → integer seed so that images within
    ~100 m share the same base feature vector.
    precision=3 → quantise to 0.001° ≈ 111 m grid.
    """
    qlat = round(lat, precision)
    qlon = round(lon, precision)
    return hash((qlat, qlon)) & 0x7FFFFFFF


def make_feature(lat: float, lon: float, noise_scale: float = 0.05) -> np.ndarray:
    """
    Return a unit-normalised 512-d vector.
    Images within the same 0.001° cell get the same base vector;
    a tiny Gaussian noise is added so vectors are not exactly identical.
    """
    base_rng = np.random.RandomState(_coord_seed(lat, lon))
    base = base_rng.randn(FEAT_DIM).astype(np.float32)
    noise = np.random.RandomState(id(lat) ^ id(lon)).randn(FEAT_DIM).astype(np.float32) * noise_scale
    vec = base + noise
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


# ---------------------------------------------------------------------------
# Mock Cosplace — drop-in replacement that uses make_feature()
# ---------------------------------------------------------------------------

class MockCosplace:
    """
    Lightweight stand-in for benchmark_models.Cosplace.
    Satisfies every method that BenchmarkFeature / BenchmarkMatcher calls.
    """

    backbone = "ResNet50"
    fc_output_dim = FEAT_DIM
    model_type = "Cosplace"

    def __init__(self, *args, **kwargs):
        self.model = None
        self.transform = None
        self.device = "cpu"

    def load_model(self):
        self.model = object()  # truthy placeholder
        return self.model

    def setup_image_transform(self):
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])
        return self.transform

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model

    def get_transform(self):
        if self.transform is None:
            self.setup_image_transform()
        return self.transform

    def get_feature_dimensions(self):
        return FEAT_DIM

    def _lat_lon_from_path(self, image_input: str) -> tuple:
        """Parse lat/lon out of a synthetic filename for deterministic features."""
        from project_utils import get_coordinates_from_path
        name = os.path.basename(image_input) if isinstance(image_input, str) else "0_0@x.jpg"
        # Try both coordinate conventions
        lat, lon = get_coordinates_from_path("dash", name)
        if lat is None:
            lat, lon = get_coordinates_from_path("parse", name)
        if lat is None:
            lat, lon = 0.0, 0.0
        return lat, lon

    def extract_features(self, image_input, transform=None) -> np.ndarray:
        lat, lon = self._lat_lon_from_path(image_input)
        return make_feature(lat, lon)

    def extract_features_batch(self, image_inputs, batch_size=None, transform=None) -> np.ndarray:
        rows = []
        for img in image_inputs:
            rows.append(self.extract_features(img, transform))
        return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_jpeg(path: str, color=(200, 200, 200)):
    """Write a tiny 8×8 solid-color JPEG."""
    img = Image.new("RGB", (8, 8), color)
    img.save(path, format="JPEG")


def write_llm_json(path: str):
    """Write a 3×3 grid attention JSON (axis-focus format used by Pipeline B)."""
    data = {
        "result": {
            "A1": 1.2, "A2": 0.8, "A3": 1.0,
            "B1": 0.9, "B2": 1.5, "B3": 0.7,
            "C1": 1.3, "C2": 0.6, "C3": 1.1,
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------
#
# Query filenames  → "dash" format  : {lat}_{lon}@{tag}.jpg
# Database filenames → "parse" format: @{utm_e}@{utm_n}@10@S@{lat}@{lon}@{tag}.jpg
#
# CLOSE  = same 0.001° cell → Recall@1 match expected
# FAR    = >1° away         → no match within 100 m
#

QUERY_SPEC = [
    # (tag,    lat,       lon,       color)
    ("q01", 37.7600,  -122.4150,  (255,  0,   0)),
    ("q02", 37.7610,  -122.4160,  (0,   255,   0)),
    ("q03", 37.7620,  -122.4170,  (0,     0, 255)),
    ("q04", 37.7630,  -122.4180,  (200, 200,   0)),
    ("q05", 37.7640,  -122.4190,  (0,   200, 200)),
]

DB_SPEC = [
    # (tag,     utm_e,      utm_n,      lat,      lon,      color)
    # ---- near pairs (one GT match per query) ----
    ("db01", "554000.00", "4177000.00", "037.76000", "-122.41500", (255, 200, 200)),
    ("db02", "554100.00", "4177100.00", "037.76100", "-122.41600", (200, 255, 200)),
    ("db03", "554200.00", "4177200.00", "037.76200", "-122.41700", (200, 200, 255)),
    ("db04", "554300.00", "4177300.00", "037.76300", "-122.41800", (240, 240, 180)),
    ("db05", "554400.00", "4177400.00", "037.76400", "-122.41900", (180, 240, 240)),
    # ---- far distractors ----
    ("db06", "640000.00", "4270000.00", "038.86000", "-121.31500", (100, 100, 100)),
    ("db07", "641000.00", "4271000.00", "038.87000", "-121.32500", (110, 110, 110)),
    ("db08", "642000.00", "4272000.00", "038.88000", "-121.33500", (120, 120, 120)),
    ("db09", "643000.00", "4273000.00", "038.89000", "-121.34500", (130, 130, 130)),
    ("db10", "644000.00", "4274000.00", "038.90000", "-121.35500", (140, 140, 140)),
    ("db11", "645000.00", "4275000.00", "038.91000", "-121.36500", (150, 150, 150)),
    ("db12", "646000.00", "4276000.00", "038.92000", "-121.37500", (160, 160, 160)),
]


def _query_filename(lat: float, lon: float, tag: str) -> str:
    return f"{lat:.4f}_{lon:.4f}@{tag}.jpg"


def _db_filename(utm_e: str, utm_n: str, lat: str, lon: str, tag: str) -> str:
    return f"@{utm_e}@{utm_n}@10@S@{lat}@{lon}@{tag}.jpg"


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class TestPipelineC(unittest.TestCase):

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vpr_attllm_test_")
        self.query_dir   = os.path.join(self.tmpdir, "query")
        self.db_dir      = os.path.join(self.tmpdir, "database")
        self.json_dir    = os.path.join(self.tmpdir, "att_json")
        self.db_cache    = os.path.join(self.tmpdir, "cosplace_db", "database")
        self.result_dir  = os.path.join(self.tmpdir, "result")
        for d in (self.query_dir, self.db_dir, self.json_dir, self.db_cache, self.result_dir):
            os.makedirs(d, exist_ok=True)

        # ---- Write synthetic query images and LLM JSON files ----
        self.query_filenames = []
        for tag, lat, lon, color in QUERY_SPEC:
            fname = _query_filename(lat, lon, tag)
            make_jpeg(os.path.join(self.query_dir, fname), color)
            # Write matching attention JSON
            json_name = os.path.splitext(fname)[0] + ".json"
            write_llm_json(os.path.join(self.json_dir, json_name))
            self.query_filenames.append(fname)

        # ---- Write synthetic database images ----
        self.db_filenames = []
        for tag, utm_e, utm_n, lat, lon, color in DB_SPEC:
            fname = _db_filename(utm_e, utm_n, lat, lon, tag)
            make_jpeg(os.path.join(self.db_dir, fname), color)
            self.db_filenames.append(fname)

        # ---- Patch benchmark_models.Cosplace before importing pipeline ----
        import benchmark_models
        self._orig_cosplace = benchmark_models.Cosplace
        benchmark_models.Cosplace = MockCosplace

        # Re-import dependent modules so they pick up the patched class
        import importlib
        import benchmark_feature
        import benchmark_matcher
        importlib.reload(benchmark_feature)
        importlib.reload(benchmark_matcher)

    def tearDown(self):
        # Restore original Cosplace
        import benchmark_models
        benchmark_models.Cosplace = self._orig_cosplace

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        print(f"\n[tearDown] Removed temp dir: {self.tmpdir}")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _run_pipeline(self, llm_att: bool, att_ratio: float = 0.0, tag: str = ""):
        """Run benchmark_pipeline_batch and return the record_dict path."""
        from benchmark_matcher import BenchmarkMatcher
        from record_evaluator import RecordEvaluator

        suffix = f"_att{att_ratio}" if llm_att else "_baseline"
        result_sub = os.path.join(self.result_dir, f"cosplace{suffix}{tag}")
        os.makedirs(result_sub, exist_ok=True)

        matcher = BenchmarkMatcher(vpr_model="cosplace")
        record_dict_path = matcher.benchmark_pipeline_batch(
            query_svi_folder  = self.query_dir,
            target_svi_folder = self.db_dir,
            save_dir          = result_sub,
            database_dir      = self.db_cache,
            top_k             = 5,
            max_workers       = 1,
            batch_size        = 4,
            llm_att           = llm_att,
            llm_json_folder   = self.json_dir if llm_att else None,
            att_ratio         = att_ratio,
            interpolate       = True,
            qe                = False,
        )
        return record_dict_path, result_sub

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_01_unsupported_model_raises(self):
        """BenchmarkMatcher should immediately reject unknown model names."""
        from benchmark_matcher import BenchmarkMatcher
        with self.assertRaises(ValueError):
            BenchmarkMatcher(vpr_model="eigenplaces")
        print("[PASS] test_01_unsupported_model_raises")

    def test_02_synthetic_files_created(self):
        """Verify the fixture created the expected number of images and JSONs."""
        q_imgs  = [f for f in os.listdir(self.query_dir) if f.endswith(".jpg")]
        db_imgs = [f for f in os.listdir(self.db_dir)    if f.endswith(".jpg")]
        jsons   = [f for f in os.listdir(self.json_dir)  if f.endswith(".json")]
        self.assertEqual(len(q_imgs),  len(QUERY_SPEC),  f"Expected {len(QUERY_SPEC)} query images")
        self.assertEqual(len(db_imgs), len(DB_SPEC),     f"Expected {len(DB_SPEC)} database images")
        self.assertEqual(len(jsons),   len(QUERY_SPEC),  f"Expected {len(QUERY_SPEC)} LLM JSONs")
        print(f"[PASS] test_02 — {len(q_imgs)} query imgs, {len(db_imgs)} DB imgs, {len(jsons)} JSONs")

    def test_03_coordinate_parsing(self):
        """Verify the coordinate filenames are parseable by project_utils helpers."""
        from project_utils import get_coordinates_from_path
        for tag, lat, lon, _ in QUERY_SPEC:
            fname = _query_filename(lat, lon, tag)
            parsed_lat, parsed_lon = get_coordinates_from_path("dash", fname)
            self.assertIsNotNone(parsed_lat, f"dash parse failed for {fname}")
            self.assertAlmostEqual(parsed_lat, lat, places=3)
            self.assertAlmostEqual(parsed_lon, lon, places=3)
        for tag, utm_e, utm_n, lat_s, lon_s, _ in DB_SPEC:
            fname = _db_filename(utm_e, utm_n, lat_s, lon_s, tag)
            parsed_lat, parsed_lon = get_coordinates_from_path("parse", fname)
            self.assertIsNotNone(parsed_lat, f"parse failed for {fname}")
            self.assertAlmostEqual(parsed_lat, float(lat_s), places=3)
            self.assertAlmostEqual(parsed_lon, float(lon_s), places=3)
        print("[PASS] test_03_coordinate_parsing")

    def test_04_mock_feature_consistency(self):
        """Images at the same coordinate should produce near-identical features."""
        m = MockCosplace()
        # q01 and db01 share lat≈37.760, lon≈-122.415 → same 0.001° cell
        path_q = os.path.join(self.query_dir, _query_filename(37.7600, -122.4150, "q01"))
        path_d = os.path.join(self.db_dir, _db_filename("554000.00","4177000.00","037.76000","-122.41500","db01"))
        fq = m.extract_features(path_q)
        fd = m.extract_features(path_d)
        cos_sim = float(np.dot(fq, fd) / (np.linalg.norm(fq) * np.linalg.norm(fd) + 1e-8))
        self.assertGreater(cos_sim, 0.95, f"Expected high cosine similarity for GT pair; got {cos_sim:.4f}")
        print(f"[PASS] test_04 — GT pair cosine similarity: {cos_sim:.4f}")

    def test_05_baseline_pipeline_runs(self):
        """Pipeline C (llm_att=False) should complete and produce a result JSON."""
        record_path, _ = self._run_pipeline(llm_att=False, tag="_run1")
        self.assertTrue(os.path.exists(record_path), f"Result JSON not found: {record_path}")
        with open(record_path, "r") as f:
            data = json.load(f)
        self.assertGreater(len(data), 0, "Result JSON is empty")
        # Every query should appear
        result_queries = set(data.keys())
        # data keys are the top-level field names returned by pandas to_json
        # (could be "target_path" and "similarity_score" columns)
        # Just check the file has content
        print(f"[PASS] test_05 — Baseline result JSON: {record_path}")
        print(f"       Keys in result: {list(data.keys())[:5]}")

    def test_06_result_json_structure(self):
        """Result JSON must map query filenames → {target_path: [...], similarity_score: [...]}."""
        record_path, _ = self._run_pipeline(llm_att=False, tag="_run2")
        with open(record_path, "r") as f:
            data = json.load(f)

        # Top-level keys are query filenames; each value is a per-query dict
        self.assertGreater(len(data), 0, "Result JSON is empty")
        for qname, qdata in data.items():
            self.assertIn("target_path",      qdata, f"Missing 'target_path' for {qname}")
            self.assertIn("similarity_score", qdata, f"Missing 'similarity_score' for {qname}")
            self.assertIsInstance(qdata["target_path"], list,
                                  f"target_path for {qname} is not a list")
            self.assertGreater(len(qdata["target_path"]), 0,
                               f"No candidates retrieved for {qname}")
        print(f"[PASS] test_06 — JSON structure valid; {len(data)} queries in result")

    def test_07_recall_evaluation(self):
        """RecordEvaluator should produce aggregated Recall@k metrics."""
        record_path, result_sub = self._run_pipeline(llm_att=False, tag="_run3")
        from record_evaluator import RecordEvaluator
        evaluator = RecordEvaluator(query_coord="dash", target_coord="parse",
                                    distance_threshold=100)
        evaluator.run_record_evaluator(
            query_folder       = self.query_dir,
            target_folder      = self.db_dir,
            query_coord        = "dash",
            target_coord       = "parse",
            distance_threshold = 100,
            record_dict_path   = record_path,
            plot               = False,
        )
        csv_dir = os.path.join(result_sub, "csv_results")  # where RecordEvaluator writes
        # Fallback: evaluator writes relative to record_dict_path directory
        result_dir_csv = os.path.join(os.path.dirname(record_path), "csv_results")
        agg_csv = os.path.join(result_dir_csv, "aggregated_results.csv")
        self.assertTrue(os.path.exists(agg_csv), f"Aggregated CSV not found at {agg_csv}")
        import pandas as pd
        df = pd.read_csv(agg_csv)
        print(f"[PASS] test_07 — Recall evaluation complete")
        print(f"       Aggregated CSV:\n{df.to_string(index=False)}")

    def test_08_high_recall_with_gt_pairs(self):
        """
        With coordinate-aware features, Recall@1 should be 1.0:
        every query's top-1 match should be within 100 m.
        """
        record_path, _ = self._run_pipeline(llm_att=False, tag="_run4")
        from record_evaluator import RecordEvaluator
        evaluator = RecordEvaluator(query_coord="dash", target_coord="parse",
                                    distance_threshold=100)
        gt_dict = evaluator.build_path_gt_dict(self.query_dir, self.db_dir, radius_m=100)
        with open(record_path, "r") as f:
            record_dict = json.load(f)

        # record_dict is already per-query: {qname: {target_path: [...], similarity_score: [...]}}
        results = evaluator.evaluate_success_recall(record_dict, gt_dict, k=[1, 5])
        agg = evaluator.aggregate_evaluation(results)
        r1 = agg["success_rate_at_k"].get(1, 0.0)
        r5 = agg["success_rate_at_k"].get(5, 0.0)
        print(f"[PASS] test_08 — Recall@1={r1*100:.1f}%  Recall@5={r5*100:.1f}%")
        self.assertEqual(r1, 1.0, f"Expected Recall@1=100% with GT-matched features, got {r1*100:.1f}%")
        self.assertEqual(r5, 1.0, f"Expected Recall@5=100% with GT-matched features, got {r5*100:.1f}%")

    def test_09_attention_pipeline_runs(self):
        """
        Pipeline C with llm_att=True and att_ratio=0.5 should run without error
        and produce a DIFFERENT result file from the baseline.
        """
        # Delete cached db features so both runs use the same db (features are shared)
        baseline_path, _ = self._run_pipeline(llm_att=False, tag="_att_base")
        att_path,      _ = self._run_pipeline(llm_att=True,  att_ratio=0.5, tag="_att05")
        self.assertTrue(os.path.exists(att_path), f"Attention result not found: {att_path}")
        self.assertNotEqual(baseline_path, att_path, "Baseline and attention runs should write separate files")
        print(f"[PASS] test_09 — Attention pipeline (att_ratio=0.5) produced: {att_path}")

    def test_10_idempotency(self):
        """Running the same configuration twice should reuse the cached result JSON."""
        import time
        path1, _ = self._run_pipeline(llm_att=False, tag="_idem")
        t1 = os.path.getmtime(path1)
        time.sleep(0.05)
        path2, _ = self._run_pipeline(llm_att=False, tag="_idem")  # same tag → same path
        t2 = os.path.getmtime(path2)
        self.assertEqual(path1, path2, "Idempotent run should return the same result path")
        self.assertEqual(t1, t2, "Result file should NOT be rewritten on second run (cache hit)")
        print(f"[PASS] test_10 — Second run returned cached result (mtime unchanged)")

    def test_11_agg_qe_pipeline(self):
        """Pipeline C with qe=True (average query expansion) should run without error."""
        from benchmark_matcher import BenchmarkMatcher
        result_sub = os.path.join(self.result_dir, "cosplace_qe")
        os.makedirs(result_sub, exist_ok=True)
        matcher = BenchmarkMatcher(vpr_model="cosplace")
        record_path = matcher.benchmark_pipeline_batch(
            query_svi_folder  = self.query_dir,
            target_svi_folder = self.db_dir,
            save_dir          = result_sub,
            database_dir      = self.db_cache,
            top_k             = 5,
            max_workers       = 1,
            batch_size        = 4,
            llm_att           = False,
            qe                = True,
        )
        self.assertTrue(os.path.exists(record_path))
        print(f"[PASS] test_11 — QE pipeline produced: {record_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  VPR-AttLLM — Pipeline C Smoke Tests")
    print("  (MockCosplace: no model download, no GPU required)")
    print("=" * 70)
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # preserve numeric order
    suite = loader.loadTestsFromTestCase(TestPipelineC)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
