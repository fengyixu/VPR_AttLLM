import logging
import os
import sys
import datetime
import warnings
import json
import pickle

from project_utils import safe_json_load
from svi_preprocess import ImgProcessor
from record_evaluator import RecordEvaluator
from agent_bot import QwenAgent, GeminiAgent
from prompts import PromptManager
from svi_agent_main import SviAgent
from svi_json_clean import JsonDataCleaner
from benchmark_matcher import BenchmarkMatcher

# Suppress OpenMP duplicate-library warnings on Windows
if sys.platform.startswith('win'):
    warnings.filterwarnings('ignore', message='.*OpenMP.*', category=RuntimeWarning)


def configure_logging(log_level=logging.INFO):
    """Configure logging to both a timestamped file (./log/) and stdout."""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(working_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'log_{timestamp}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    for noisy_logger in [
        "httpcore", "httpx", "urllib3", "PIL", "PIL.PngImagePlugin",
        "google_genai.models", "faiss.loader", "matplotlib", "graph_embedding",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger


logger = configure_logging(log_level=logging.INFO)


# =============================================================================
# Pipeline A — Flooding SVI Image Generation
# =============================================================================

def run_svi_image_generation_pipeline():
    """
    Use a Gemini vision-language model to generate flood-simulated versions of
    street-view images (SVI).  Edit the paths and prompt below to match your data.

    Prerequisites
    -------------
    - Set GEMINI_API_KEY environment variable.
    - Place source SVI images in `test_folder`.
    """
    prompt_manager = PromptManager()
    svi_agent = SviAgent(
        model_name="gemini-2.0-flash",
        bot_class=GeminiAgent,
        bot_kwargs={},  # API key is read from GEMINI_API_KEY env var
    )

    prompt_list = [prompt_manager.get_prompt("image_1", "same_view_v2.3")]

    # ---- Configure your paths here ----
    test_folder = "C:/VPR_temp/hk-urban/hk_mapi"
    output_folder = os.path.join(os.path.dirname(test_folder),
                                 f"{os.path.basename(test_folder)}_flooding")

    if os.path.exists(test_folder):
        svi_agent.svi_generation_batch_pipeline(
            svi_folder=test_folder,
            prompt_list=prompt_list,
            image_save_folder=output_folder,
            max_workers=12,
            retry=False,
        )
    else:
        print(f"Test folder not found: {test_folder}")


# =============================================================================
# Pipeline B — LLM Attention Annotation
# =============================================================================

def run_svi_llm_attention_pipeline():
    """
    Send street-view images to a vision-language model and obtain per-region
    attention weights (grid JSON).  The resulting JSON files are later consumed
    by AttentionCosPlace during retrieval.

    Prerequisites
    -------------
    - Set GEMINI_API_KEY (or DASHSCOPE_API_KEY for Qwen) environment variable.
    - Place query images in `svi_folder`.
    """
    # --- Gemini (recommended) ---
    MODEL_NAME = "gemini-2.5-flash"
    svi_agent = SviAgent(
        model_name=MODEL_NAME,
        bot_class=GeminiAgent,
        bot_kwargs={},  # API key from GEMINI_API_KEY env var
    )

    # --- Qwen alternative (uncomment to use) ---
    # MODEL_NAME = "qwen3.5-plus"
    # svi_agent = SviAgent(
    #     model_name=MODEL_NAME,
    #     bot_class=QwenAgent,
    #     bot_kwargs={},  # API key from DASHSCOPE_API_KEY env var
    # )

    prompt_manager = PromptManager()

    # ---- Configure your dataset paths here ----
    project_dir = "C:/VPR_temp/hk-urban"
    svi_folder = os.path.join(project_dir, "hk_flood_test")
    city = "Hong Kong"

    # SF-XL example (uncomment to switch):
    # project_dir = "D:/VPR_dataset/sf-xl/processed/test"
    # svi_folder = os.path.join(project_dir, "sf_flood")
    # city = "San Francisco"

    preprocess_img = True
    prompt_list = [prompt_manager.get_prompt("svi_att", "axis_focus", city=city)]
    json_save_folder = os.path.join(svi_folder, f"att_axis@{svi_agent.model_name}")

    svi_agent.svi_attention_batch_pipeline(
        svi_folder, prompt_list, json_save_folder,
        preprocess_img=preprocess_img,
        grid_num=0,
        grid_line=False,
        axis=True,
        max_workers=100,
        retry=True,
    )

    cleaner = JsonDataCleaner()
    cleaner.clean_att_content_batch_pipeline(json_save_folder)
    cleaner.remove_failed_file(json_save_folder)


def run_llm_json_clean_pipeline():
    """Re-run JSON cleaning on an existing attention folder (repair-only pass)."""
    check_json_folder = "D:/VPR_dataset/sf-xl/processed/test/sf_flood/att_axis_minimal@qwen3.5-plus"
    cleaner = JsonDataCleaner()
    print(f"Cleaning: {check_json_folder}")
    cleaner.remove_failed_file(check_json_folder)


# =============================================================================
# Pipeline C — VPR Matching and Evaluation
# =============================================================================

def run_llm_match_pipeline(
    att_ratio: float,
    vpr_model: str,
    llm_attention: bool,
    interpolate: bool,
    qe: bool,
    auto_execute: bool,
    working_dir: str,
    query_svi_folder: str,
    target_svi_folder: str,
    query_coord: str,
    target_coord: str,
    llm_json_folder: str,
    distance_threshold: int,
    evaluator: RecordEvaluator | None,
    matcher: BenchmarkMatcher | None,
):
    """
    Run a single VPR matching + evaluation pass.

    Args
    ----
    att_ratio         : LLM attention blend factor α ∈ [0, 1].
                        0.0 → standard CosPlace;  1.0 → full LLM attention.
    vpr_model         : Must be "cosplace".
    llm_attention     : Enable AttentionCosPlace attention injection.
    interpolate       : Bilinear-interpolate the LLM grid attention map.
    qe                : Apply Average Query Expansion after first retrieval.
    auto_execute      : Skip the interactive confirmation prompt.
    working_dir       : Root directory containing query and database sub-folders.
    query_svi_folder  : Absolute path to query images.
    target_svi_folder : Absolute path to database images.
    query_coord       : Coordinate format for query filenames ("dash" or "parse").
    target_coord      : Coordinate format for database filenames ("dash" or "parse").
    llm_json_folder   : Folder with per-image LLM attention JSON files.
    distance_threshold: Positive-match radius in metres.
    evaluator         : Reuse a pre-built RecordEvaluator (None to create a new one).
    matcher           : Reuse a pre-loaded BenchmarkMatcher (None to create a new one).

    Returns
    -------
    dict with keys "result_save_dir" and "record_dict_path", or None if skipped.
    """
    database_dir = os.path.join(working_dir, f"{vpr_model}_db",
                                os.path.basename(target_svi_folder))
    if llm_attention:
        result_save_dir = os.path.join(
            os.getcwd(), "result",
            f"{vpr_model}_llmatt_feature_target_axis",
            f"{os.path.basename(query_svi_folder)}@{os.path.basename(target_svi_folder)}_{att_ratio}",
        )
    else:
        result_save_dir = os.path.join(
            os.getcwd(), "result",
            f"{vpr_model}",
            f"{os.path.basename(query_svi_folder)}@{os.path.basename(target_svi_folder)}",
        )

    print(f"Model        : {vpr_model}")
    print(f"Database dir : {database_dir}")
    print(f"Result dir   : {result_save_dir}")
    print(f"LLM attention: {llm_attention}  interpolate: {interpolate}  att_ratio: {att_ratio}")
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(database_dir, exist_ok=True)

    if evaluator is None:
        evaluator = RecordEvaluator(query_coord=query_coord,
                                    target_coord=target_coord,
                                    distance_threshold=distance_threshold)
    if matcher is None:
        matcher = BenchmarkMatcher(vpr_model=vpr_model)

    if not auto_execute:
        answer = input("Execute the benchmark pipeline? (y/n): ")
        if answer.strip().lower() == "n":
            return None

    record_dict_path = matcher.benchmark_pipeline_batch(
        query_svi_folder,
        target_svi_folder,
        result_save_dir,
        database_dir,
        top_k=100,
        max_workers=1,       # Always 1 for GPU to avoid memory conflicts
        batch_size=32,
        llm_att=llm_attention,
        llm_json_folder=llm_json_folder,
        rerank=False,
        att_ratio=att_ratio,
        interpolate=interpolate,
        qe=qe,
    )

    evaluator.run_record_evaluator(
        query_svi_folder,
        target_svi_folder,
        query_coord,
        target_coord,
        distance_threshold,
        record_dict_path,
        plot=False,
    )

    return {"result_save_dir": result_save_dir, "record_dict_path": record_dict_path}


def run_llm_match_sweep(att_ratio_list=(0.0, 0.5, 1.0)):
    """
    Convenience wrapper that runs run_llm_match_pipeline() for multiple att_ratio
    values while reusing the same loaded model and evaluator.

    Edit the configuration variables below to point to your dataset.
    """
    # ---- Configuration ----
    vpr_model = "cosplace"
    llm_attention = True
    interpolate = True
    qe = False

    working_dir = "D:/VPR_dataset/sf-xl/processed/test"

    query_svi_folder = os.path.join(working_dir, "sf_flood")
    target_svi_folder = os.path.join(working_dir, "database_full")
    query_coord = "dash"
    target_coord = "parse"
    distance_threshold = 100  # metres

    # LLM attention JSON folder (choose one):
    # Gemini:
    # llm_json_folder = os.path.join(query_svi_folder, "att_axis@gemini-2.5-flash")
    # Qwen:
    # llm_json_folder = os.path.join(query_svi_folder, "att_axis@qwen3.5-plus")
    # Qwen3-vl-8b (local):
    llm_json_folder = os.path.join(query_svi_folder, "att_axis_minimal@qwen3-vl-8b")

    # Pre-build heavy objects once and reuse across ratio sweep
    dummy_eval = RecordEvaluator(query_coord=query_coord,
                                  target_coord=target_coord,
                                  distance_threshold=distance_threshold)
    dummy_matcher = BenchmarkMatcher(vpr_model=vpr_model)

    results = []
    for ratio in att_ratio_list:
        out = run_llm_match_pipeline(
            att_ratio=ratio,
            vpr_model=vpr_model,
            llm_attention=llm_attention,
            interpolate=interpolate,
            working_dir=working_dir,
            query_svi_folder=query_svi_folder,
            target_svi_folder=target_svi_folder,
            query_coord=query_coord,
            target_coord=target_coord,
            distance_threshold=distance_threshold,
            llm_json_folder=llm_json_folder,
            auto_execute=True,
            evaluator=dummy_eval,
            matcher=dummy_matcher,
            qe=qe,
        )
        results.append(out)

    return results


def run_evaluate_pipeline():
    """Stand-alone evaluation pass over a pre-computed result JSON."""
    working_dir = r"D:\VPR_dataset\sf-xl\processed\test"
    query_svi_folder = os.path.join(working_dir, "sf_flood")
    target_svi_folder = os.path.join(working_dir, "database_full")
    query_coord = "dash"
    target_coord = "parse"
    distance_threshold = 100

    record_dict_path = r"D:\VPR_dataset\sf-xl\result\cosplace\sf_flood@database_full\cosplace_results.json"

    evaluator = RecordEvaluator(query_coord=query_coord,
                                target_coord=target_coord,
                                distance_threshold=distance_threshold)
    answer = input("Execute the evaluation pipeline? (y/n): ")
    if answer.strip().lower() == "n":
        return
    evaluator.run_record_evaluator(
        query_svi_folder,
        target_svi_folder,
        query_coord,
        target_coord,
        distance_threshold,
        record_dict_path,
        plot=False,
    )


# =============================================================================
# Entry point
# =============================================================================

def main():
    # ---- Uncomment the pipeline you want to run ----

    # Pipeline A: Generate flooding SVI images
    # run_svi_image_generation_pipeline()

    # Pipeline B: Generate LLM attention annotations
    # run_svi_llm_attention_pipeline()

    # Pipeline C: VPR matching + evaluation (sweep over att_ratio values)
    run_llm_match_sweep(att_ratio_list=(0.0, 0.5, 1.0))


if __name__ == "__main__":
    main()
