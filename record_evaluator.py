from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from project_utils import get_coordinates_from_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class RecordEvaluator:
    def __init__(self, query_coord="dash", target_coord="parse", distance_threshold=25):
        self.query_coord = query_coord
        self.target_coord = target_coord
        self.logger = logger
        self.distance_threshold = distance_threshold

    def build_path_gt_dict(self, query_folder, target_folder, radius_m=25, save_path=None):
        """
        Build GT dict with file paths directly.

        Returns:
            dict: {query_path: [gt_target_paths]}
        """
        if save_path is None:
            query_folder_basename = os.path.basename(query_folder)
            save_path = os.path.join(os.path.dirname(query_folder), f'{query_folder_basename}@gt_{radius_m}m.pkl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                gt_dict = pickle.load(f)
            print(f"GT dict loaded from {save_path}")
            return gt_dict
        else:
            print(f"Building gt_dict...")

        valid_exts = ('.json', '.jpg', '.jpeg', '.png')
        # Get paths and coordinates
        query_paths = sorted(Path(query_folder).glob('*'))
        query_paths = [p for p in query_paths if p.suffix in valid_exts]
        # Target paths: prefer list file if available, else recurse
        list_file = os.path.join(target_folder, 'database_images_paths.txt')
        if os.path.exists(list_file):
            print("Building gt from path text")
            target_paths = []
            with open(list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    if not any(rel_path.lower().endswith(ext) for ext in valid_exts):
                        continue
                    target_paths.append(Path(os.path.join(target_folder, rel_path)))
        else:
            print("Building gt from files within folder")
            target_paths = sorted(Path(target_folder).glob('*'))
            target_paths = [p for p in target_paths if p.suffix in valid_exts]

        query_coords = np.array([get_coordinates_from_path(self.query_coord, p) for p in query_paths])
        target_coords = np.array([get_coordinates_from_path(self.target_coord, p) for p in target_paths])

        # Build spatial index
        nbrs = NearestNeighbors(algorithm='ball_tree', metric='haversine')
        nbrs.fit(np.radians(target_coords))

        # Query all
        radius_rad = (radius_m / 1000) / 6371.0
        distances, indices = nbrs.radius_neighbors(np.radians(query_coords), radius=radius_rad)

        # Build path-based GT dict using basenames as keys
        gt_dict = {}
        for i, query_path in enumerate(query_paths):
            gt_paths = [os.path.basename(str(target_paths[idx])) for idx in indices[i]]
            query_basename = os.path.basename(str(query_path))
            gt_dict[query_basename] = gt_paths

        with open(save_path, 'wb') as f:
            pickle.dump(gt_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"GT dict saved to {save_path}")

        return gt_dict

    def evaluate_success_recall(self, record_dict, gt_dict, k=[1,5,10]):
        """
        Fast success recall evaluation for each query within the record_dict.
        Return the test results in a list, each item is a dict with the following keys:
        - query_path: the query path
        - predicted_paths: the predicted paths, following the record_dict order
        - gt_paths: the ground truth paths, following the gt_dict order
        - success_at_1: the success at 1
        - success_at_5: the success at 5
        - success_at_10: the success at 10
        """
        test_results = []
        
        for query_path, query_data in record_dict.items():
            # Ensure we compare basenames, since GT dict uses basenames as keys
            query_basename = os.path.basename(query_path)
            if query_basename in gt_dict:
                # print(query_basename)
                # Extract predicted paths from the nested structure
                predicted_paths = query_data.get("target_path", [])
                # Convert predicted paths to basenames for comparison
                predicted_basenames = [os.path.basename(p) for p in predicted_paths]
                gt_paths = gt_dict[query_basename]
                gt_paths_set = set(gt_paths)
                # Calculate success at each k
                success_at_k = {}
                for k_val in k:
                    predicted_set = set(predicted_basenames[:k_val])
                    intersection = gt_paths_set & predicted_set
                    success_at_k[k_val] = 1 if intersection else 0
                    
                
                result = {
                    'query_path': query_path,
                    'predicted_paths': predicted_paths,
                    'gt_paths': gt_paths,
                    'success_at_k': success_at_k
                }
                test_results.append(result)
        
        return test_results

    def aggregate_evaluation(self, test_results):
        """
        Aggregate evaluation results across multiple test queries.

        Args:
            test_results (list): List of evaluation dict
            
        Returns:
            dict: Aggregated metrics
        """

        # Determine available k values from success_at_k
        available_k = set()
        for r in test_results:
            sak = r.get('success_at_k', {})
            for k in sak.keys():
                available_k.add(k)
        if not available_k:
            # Fallback for legacy results
            available_k = {1, 5, 10}
        available_k = sorted(list(available_k))

        # Filter out evaluations that have at least success@1 computed
        successful_results = [r for r in test_results if 'success_at_k' in r]
        num_successful = len(successful_results)

        if num_successful == 0:
            return {"error": "No successful evaluations found"}

        # Calculate aggregated metrics
        aggregated = {
            'total_queries': len(test_results),
            'evaluated_queries': num_successful,
            'distance_threshold_m': self.distance_threshold,
            'success_rate_at_k': {}
        }

        for k in available_k:
            vals = [r.get('success_at_k', {}).get(k, 0) for r in successful_results]
            aggregated['success_rate_at_k'][k] = float(sum(vals)) / len(test_results) if len(test_results) > 0 else 0.0

        print(f"total queries: {len(test_results)}")
        for k in available_k:
            rate = round(aggregated['success_rate_at_k'][k] * 100, 2)
            print(f"success@{k}: {rate}%")

        return aggregated

    def plot_result(self, test_result, query_folder, target_folder, save_dir, max_k=10):
        """
        Plot the visualization data for the given test result. The plot will be saved in the save_dir.
            - the query image will be plotted in the first row, from the left
            - the gt images will be plotted in the second row, from the left
            - the predicted images will be plotted in the third row, from the left following test result order
                - true predicted images will be plotted with green border
                - false predicted images will be plotted with red border

        Args:
            test_result: the test result for a single query
            query_folder: the query svi folder, containing the query images
            target_folder: the target svi folder, containing the target images
            save_dir: the directory to save the plot, base name will be the os.basename (query_path) (remove the extension)
            max_k: maximum number of predicted images to plot (default: 10)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        query_path = test_result['query_path']
        predicted_paths = test_result['predicted_paths'][:max_k]  # Limit to top-k
        gt_paths = test_result['gt_paths']  # Use full GT paths for correct comparison
        
        # Get base name for saving
        base_name = os.path.splitext(os.path.basename(query_path))[0]
        
        # Load query image
        query_img_path = os.path.join(query_folder, os.path.basename(query_path))
        if not os.path.exists(query_img_path):
            query_img_path = query_path
        
        try:
            query_img = Image.open(query_img_path)
        except Exception as e:
            print(f"ERROR: Could not load query image: {query_img_path}, error: {e}")
            return
        
        # Calculate grid dimensions - use max_k for display, but show all GT paths that exist
        max_cols = max(min(len(gt_paths), max_k), len(predicted_paths), 1)
        rows = 3  # query, gt, predicted
        cols = max_cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot query image
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query', fontsize=10)
        axes[0, 0].axis('off')
        
        # Hide remaining cells in query row
        for j in range(1, cols):
            axes[0, j].axis('off')
        
        # Plot GT images - use full gt_paths for comparison, display up to max_k
        gt_paths_set = set(gt_paths)  # Full GT paths for correct comparison
        gt_paths_display = gt_paths[:max_k]  # Limit display to max_k for visualization
        
        for j, gt_path in enumerate(gt_paths_display):
            if j >= cols:
                break
            # gt_path is already a basename, so construct full path
            gt_img_path = os.path.join(target_folder, gt_path)
            if not os.path.exists(gt_img_path):
                gt_img_path = gt_path
            
            try:
                gt_img = Image.open(gt_img_path)
                axes[1, j].imshow(gt_img)
                axes[1, j].set_title(f'GT {j+1}', fontsize=10)
            except Exception as e:
                print(f"ERROR: Could not load GT image {j+1}: {gt_img_path}, error: {e}")
                axes[1, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[1, j].set_title(f'GT {j+1}', fontsize=10)
            axes[1, j].axis('off')
        
        # Hide remaining cells in GT row
        for j in range(len(gt_paths_display), cols):
            axes[1, j].axis('off')
        
        # Plot predicted images
        for j, pred_path in enumerate(predicted_paths):
            if j >= cols:
                break
            pred_img_path = os.path.join(target_folder, os.path.basename(pred_path))
            if not os.path.exists(pred_img_path):
                pred_img_path = pred_path
            
            try:
                pred_img = Image.open(pred_img_path)
                axes[2, j].imshow(pred_img)
                
                # Color border based on correctness - compare basenames
                pred_basename = os.path.basename(pred_path)
                is_correct = pred_basename in gt_paths_set
                border_color = 'green' if is_correct else 'red'
                border_width = 5  # Thicker border
                
                # Add border by creating a new image with border
                bordered_img = Image.new('RGB', 
                    (pred_img.width + 2*border_width, pred_img.height + 2*border_width), 
                    border_color)
                bordered_img.paste(pred_img, (border_width, border_width))
                axes[2, j].imshow(bordered_img)
                
                axes[2, j].set_title(f'Pred {j+1}', fontsize=10)
            except Exception as e:
                print(f"ERROR: Could not load predicted image {j+1}: {pred_img_path}, error: {e}")
                axes[2, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[2, j].set_title(f'Pred {j+1}', fontsize=10)
            axes[2, j].axis('off')
        
        # Hide remaining cells in predicted row
        for j in range(len(predicted_paths), cols):
            axes[2, j].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{base_name}.png')
        
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"ERROR: Failed to save plot: {e}")
            return
        
        plt.close()


    def plot_results_pipeline(self, test_results, query_folder, target_folder, plot_dir, max_k=10):
        """
        Plot results for all queries, organizing them into success and failure subfolders.
        
        Args:
            test_results: List of test results from evaluate_success_recall
            query_folder: Path to query images
            target_folder: Path to target images  
            plot_dir: Base directory for saving plots
            max_k: Maximum number of images to plot
        """
        success_dir = os.path.join(plot_dir, 'success')
        failure_dir = os.path.join(plot_dir, 'failure')
        
        success_count = 0
        failure_count = 0
        
        for i, result in tqdm(enumerate(test_results), total=len(test_results), desc="Plotting results"):
            success_at_10 = result['success_at_k'].get(10, 0)
            
            if success_at_10 == 1:
                save_dir = success_dir
                success_count += 1
            else:
                save_dir = failure_dir
                failure_count += 1
            
            self.plot_result(result, query_folder, target_folder, save_dir, max_k)
        
        print(f"Pipeline complete: {success_count} success cases, {failure_count} failure cases")
        print(f"Success plots saved to: {success_dir}")
        print(f"Failure plots saved to: {failure_dir}")

    def save_results(self, test_results, aggregated_results, save_dir):
        """
        Save test results and aggregated results to CSV files.
        
        Args:
            test_results: List of test results from evaluate_success_recall
            aggregated_results: Aggregated results from aggregate_evaluation
            save_dir: Directory to save the CSV files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare individual test results for CSV
        individual_data = []
        for result in test_results:
            row = {
                'query_path': result['query_path'],
                'num_predicted_paths': len(result['predicted_paths']),
                'num_gt_paths': len(result['gt_paths']),
                'predicted_paths': ';'.join(result['predicted_paths']),  # Join with semicolon
                'gt_paths': ';'.join(result['gt_paths'])  # Join with semicolon
            }
            
            # Add success_at_k columns
            success_at_k = result.get('success_at_k', {})
            for k in sorted(success_at_k.keys()):
                row[f'success_at_{k}'] = success_at_k[k]
            
            individual_data.append(row)
        
        # Save individual test results
        individual_df = pd.DataFrame(individual_data)
        individual_csv_path = os.path.join(save_dir, f'individual_results.csv')
        individual_df.to_csv(individual_csv_path, index=False)
        print(f"Individual test results saved to: {individual_csv_path}")
        
        # Prepare aggregated results for CSV
        aggregated_data = []
        if 'error' not in aggregated_results:
            # Main metrics
            aggregated_data.append({
                'metric': 'total_queries',
                'value': aggregated_results['total_queries']
            })
            aggregated_data.append({
                'metric': 'evaluated_queries', 
                'value': aggregated_results['evaluated_queries']
            })
            aggregated_data.append({
                'metric': 'distance_threshold_m',
                'value': aggregated_results['distance_threshold_m']
            })
            
            # Success rates at different k values
            success_rates = aggregated_results.get('success_rate_at_k', {})
            for k in sorted(success_rates.keys()):
                aggregated_data.append({
                    'metric': f'success_rate_at_{k}',
                    'value': success_rates[k]
                })
                aggregated_data.append({
                    'metric': f'success_rate_at_{k}_percent',
                    'value': round(success_rates[k] * 100, 2)
                })
        else:
            aggregated_data.append({
                'metric': 'error',
                'value': aggregated_results['error']
            })
        
        # Save aggregated results
        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_csv_path = os.path.join(save_dir, f'aggregated_results.csv')
        aggregated_df.to_csv(aggregated_csv_path, index=False)
        print(f"Aggregated results saved to: {aggregated_csv_path}")
        
        return individual_csv_path, aggregated_csv_path

    def run_record_evaluator(self, query_folder, target_folder, query_coord, target_coord, distance_threshold, record_dict_path, plot=False):

        result_dir = os.path.dirname(record_dict_path)
        plot_dir = os.path.join(result_dir, 'plots')

        evaluator = RecordEvaluator(query_coord=query_coord, target_coord=target_coord)
        
        gt_dict = evaluator.build_path_gt_dict(query_folder, target_folder, radius_m=distance_threshold)
        # Load record dictionary from JSON
        print(f"Loading record_dict from {record_dict_path}...")
        with open(record_dict_path, 'r') as f:
            record_dict = json.load(f)
        
        # Evaluation
        test_results = evaluator.evaluate_success_recall(record_dict, gt_dict)
        aggregated = evaluator.aggregate_evaluation(test_results)
        # Save results to CSV files
        print(f"Saving results to CSV...")
        csv_dir = os.path.join(result_dir, 'csv_results')
        evaluator.save_results(test_results, aggregated, csv_dir)

        # Plot all results using pipeline
        if plot:
            print(f"Plotting all results...")
            evaluator.plot_results_pipeline(test_results, query_folder, target_folder, plot_dir, max_k=10)
        else:
            print("Skipping plotting")
