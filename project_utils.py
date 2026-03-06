import re
import json
import numpy as np
import logging
import os
from math import radians, cos, sin, asin, sqrt
# Create filtered FAISS index with only candidate features
import faiss

logger = logging.getLogger(__name__)


def dash_coordinates(filename):
    """
    Extract latitude and longitude coordinates from a filename.
    Supports formats: 
    - 'latitude_longitude_*.json' and 'latitude_longitude@*.json'
    - 'name%latitude_longitude@*.json' (extracts from % section)
    Returns (lat, lon) as floats, or (None, None) if parsing fails.
    """
    if not filename or not isinstance(filename, str):
        return None, None
    
    try:
        # Remove file extension
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Check if % exists - if so, extract the part after %
        if '%' in base_name:
            base_name = base_name.split('%', 1)[1]
        
        # Check if @ exists
        if '@' in base_name:
            # Split by @ to get coordinate part (before @)
            coord_part = base_name.split('@')[0]
        else:
            # No @, use the entire base_name
            coord_part = base_name
        
        # Split coordinates by underscore and take first two parts
        parts = coord_part.split('_')
        
        if len(parts) < 2:
            return None, None
        
        # Only use first two parts for coordinates
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None, None
        return lat, lon
        
    except (IndexError, ValueError, AttributeError):
        return None, None

# def dash_coordinates_old(filename):
#     """
#     Extract latitude and longitude coordinates from a filename.
#     Supports formats: 'latitude_longitude_*.json' and 'latitude_longitude@*.json'.
#     Returns (lat, lon) as floats, or (None, None) if parsing fails.
#     """
#     if not filename or not isinstance(filename, str):
#         return None, None
#     try:
#         base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
#         parts = re.split(r'[_@]', base_name)
#         if len(parts) < 2:
#             return None, None
#         lat = float(parts[0].strip())
#         lon = float(parts[1].strip())
#         if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
#             return None, None
#         return lat, lon
#     except (IndexError, ValueError, AttributeError):
#         return None, None

def parse_coordinates(filename):
    """For SF-XL: Extract latitude and longitude from filename."""
    name = filename.rsplit('.', 1)[0]
    parts = name.split('@')
    try:
        if len(parts) > 7:
            lat = float(parts[5].strip())
            lon = float(parts[6].strip())
            return lat, lon
        else:
            # get lat and lon for the merged file name: 0543331.40@4179325.38@10@S@037.76022@-122.50806.json
            lat = float(parts[4].strip())
            lon = float(parts[5].strip())
            return lat, lon
    except (IndexError, ValueError):
        return None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in meters using the Haversine formula."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r
   
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in meters between arrays of points (lat1, lon1) and (lat2, lon2).
    lat1, lon1: shape (N,)
    lat2, lon2: shape (M,)
    Returns: (N, M) array of distances in meters
    """
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return r * c

def get_coordinates_from_path(coord_type, file_path):
    """
    Extract coordinates from file path based on experiment type.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if extraction fails
    """
    filename = os.path.basename(file_path)
    
    if coord_type == "dash":
        return dash_coordinates(filename)
    elif coord_type == "parse":
        return parse_coordinates(filename)
    else:
        logger.warning(f"Unknown coordinate type: {coord_type}")
        return None, None

def get_utm_from_path(coord_type, file_path):
    """
    Extract UTM coordinates from file path based on experiment type.
    
    Args:
        coord_type (str): Coordinate extraction method ("dash" or "parse")
        - for dash mode, try get UTM from: f"{idx}%{lat_str}_{lon_str}@{f_code}@{n_code}@{d_code}@{utm_easting}@{utm_northing}"
        file_path (str): Path to the file
        
    Returns:
        tuple: (utm_easting, utm_northing) or (None, None) if extraction fails
    """
    filename = os.path.basename(file_path)
    # remove the file extension
    filename = filename.rsplit('.', 1)[0]
    
    if coord_type == "dash":
        # For dash format, UTM coordinates are not available
        parts = filename.split('@')
        if len(parts) >= 5:
            utm_easting = float(parts[4].strip())
            utm_northing = float(parts[5].strip())
            return utm_easting, utm_northing
        return None, None
    elif coord_type == "parse":
        # For parse format: path/to/file/@utm_easting@utm_northing@...@.jpg
        try:
            parts = filename.split('@')
            if len(parts) >= 3:
                utm_easting = float(parts[1].strip())
                utm_northing = float(parts[2].strip())
                return utm_easting, utm_northing
        except (IndexError, ValueError):
            pass
        return None, None
    else:
        logger.warning(f"Unknown coordinate type: {coord_type}")
        return None, None

def extract_place_id(filename: str, coord_type: str = "dash") -> str:
    """
    Extract place ID from filename.
    
    For 'dash' mode: Uses part before first "@"
    For 'parse' mode: Creates place_id from all components before lat@lon
    
    Args:
        filename: Image filename
        
    Returns:
        Place ID string
    """
    if coord_type == "dash":
        if '%' in filename:
            return filename.split('%')[0]
        return filename.split('@')[0]
    elif coord_type == "parse":
        # Extract all components before lat@lon
        # Format: @timestamp1@timestamp2@zoom@S@lat@lon@hash@@0@@@@date@@.jpg
        parts = filename.split('@')
        if len(parts) >= 7:
            # Include parts 1-4: timestamp1, timestamp2, zoom, S
            # Plus parts 5-6: lat, lon
            place_id_parts = parts[1:7]  # [timestamp1, timestamp2, zoom, S, lat, lon]
            
            # Rejoin with @ separator
            return '@'.join(place_id_parts)
        else:
            # Fallback to full filename if parsing fails
            # Use full file name for merged files (already place_id): 0543331.40@4179325.38@10@S@037.76022@-122.50806.json
            return filename
    else:
        raise ValueError(f"Invalid coordinate type: {coord_type}")

def safe_json_load(file_path):
    """Safely load JSON file with UTF-8 encoding to handle special characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to system default encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        return None

# Helper function to get candidate targets from initial results
def get_candidate_targets(query_filename, initial_result_dict):
    """Get candidate targets from initial results for reranking."""
    if initial_result_dict is None:
        return None, None
    
    for ext in ['.jpg', '.png', '.jpeg']:
        query_image_filename = query_filename.replace('.json', ext)
        if query_image_filename in initial_result_dict:
            return (initial_result_dict[query_image_filename]["target_path"],
                    initial_result_dict[query_image_filename]["similarity_score"])
    logger.debug(f"No initial results found for {query_filename}")
    return None, None

def get_filtered_features_and_index(query_filename, initial_result_dict, reference_features, faiss_index):
    """
    Get filtered reference features and FAISS index based on initial results for reranking.
    
    Args:
        query_filename: Query filename (e.g., "query.json")
        initial_result_dict: Dictionary containing initial results
        reference_features: Full list of reference features
        faiss_index: Full FAISS index
        
    Returns:
        tuple: (filtered_reference_features, filtered_faiss_index) or (None, None) if no candidates found
    """
    if initial_result_dict is None:
        return None, None
    
    # Get candidate targets from initial results
    candidate_targets, candidate_scores = get_candidate_targets(query_filename, initial_result_dict)
    if candidate_targets is None or not candidate_targets:
        return None, None
    
    # Create a set of candidate target filenames for fast lookup
    candidate_target_set = set()
    for target_path in candidate_targets:
        # Extract just the filename from the target path
        target_filename = os.path.basename(target_path)
        candidate_target_set.add(target_filename)
    
    # Filter reference features to only include candidates
    filtered_reference_features = []
    candidate_indices = []
    
    for idx, ref_feat in enumerate(reference_features):
        ref_filename = os.path.basename(ref_feat['path'])
        if ref_filename in candidate_target_set:
            filtered_reference_features.append(ref_feat)
            candidate_indices.append(idx)
    
    if not filtered_reference_features:
        logger.debug(f"No matching reference features found for candidates of {query_filename}")
        return None, None
        
    # Extract features from filtered reference features
    filtered_features_array = np.array([feat['features'] for feat in filtered_reference_features]).astype('float32')
    
    # Create new FAISS index with same configuration as original
    dimension = filtered_features_array.shape[1]
    filtered_faiss_index = faiss.IndexFlatL2(dimension)
    filtered_faiss_index.add(filtered_features_array)
    
    logger.debug(f"Filtered to {len(filtered_reference_features)} candidates for {query_filename}")
    return filtered_reference_features, filtered_faiss_index

# ============ LLM-att for optimized models ===============================

def llm_grid_to_attention_map(llm_dict, H_feat: int, W_feat: int, device, interpolate=False):
    """Convert grid dict (e.g., {'A1': w, ...}) to a [H_feat, W_feat] attention map tensor.
    - Supports weights in [0, 2]; values are clamped.
    - If interpolate=True, creates smooth transitions between grid cells.
    - If interpolate=False, creates sharp grid boundaries (default).
    Note: callers must pass the grid dict directly (not a wrapper with 'result').
    """
    import math
    import torch as _torch

    num_cells = len(llm_dict)
    if num_cells == 0:
        # Handle empty dict case - return uniform attention map
        return _torch.ones((H_feat, W_feat), device=device)
    
    grid_f = math.sqrt(num_cells)
    if int(grid_f) != grid_f:
        raise ValueError("llm_dict size must be a perfect square (NxN grid)")
    grid_n = int(grid_f)

    if interpolate:
        # Create interpolated attention map with smooth transitions
        # First create a low-resolution grid tensor
        grid_tensor = _torch.ones((grid_n, grid_n), device=device)
        
        def parse_cell(cell_id: str):
            row_idx = ord(cell_id[0].upper()) - ord('A')
            col_idx = int(cell_id[1:]) - 1
            return row_idx, col_idx

        for cell_id, w in llm_dict.items():
            if not isinstance(cell_id, str) or len(cell_id) < 2:
                continue
            try:
                w = float(w)
            except Exception:
                continue
            w = max(0.0, min(2.0, w))
            r, c = parse_cell(cell_id)
            if r < 0 or r >= grid_n or c < 0 or c >= grid_n:
                continue
            grid_tensor[r, c] = float(w)
        
        # Interpolate to full resolution using bilinear interpolation
        grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, grid_n, grid_n]
        att = _torch.nn.functional.interpolate(
            grid_tensor, size=(H_feat, W_feat), 
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)  # [H_feat, W_feat]
        
    else:
        # Create sharp grid boundaries (original behavior)
        att = _torch.ones((H_feat, W_feat), device=device)

        def parse_cell(cell_id: str):
            row_idx = ord(cell_id[0].upper()) - ord('A')
            col_idx = int(cell_id[1:]) - 1
            return row_idx, col_idx

        cell_h = H_feat / grid_n
        cell_w = W_feat / grid_n

        for cell_id, w in llm_dict.items():
            if not isinstance(cell_id, str) or len(cell_id) < 2:
                continue
            try:
                w = float(w)
            except Exception:
                continue
            w = max(0.0, min(2.0, w))
            r, c = parse_cell(cell_id)
            if r < 0 or r >= grid_n or c < 0 or c >= grid_n:
                continue
            y1 = int(round(r * cell_h)); y2 = H_feat if r == grid_n - 1 else int(round((r + 1) * cell_h))
            x1 = int(round(c * cell_w)); x2 = W_feat if c == grid_n - 1 else int(round((c + 1) * cell_w))
            if y2 > y1 and x2 > x1:
                att[y1:y2, x1:x2] = float(w)

    return att

# [TODO] Refine this function: only applying the weights around radius=15% of the centerpoint, and use default 1 for all other points
def llm_coord_to_attention_map(llm_dict_or_list, H_feat: int, W_feat: int, device, sigma: float = 0.2):
    """
    Convert coord-based LLM list to a [H_feat, W_feat] attention map tensor using smooth interpolation.
    - Input accepts either wrapper { 'result': [...] } or the list directly.
    - Each item: { "center": [x in 0..1, y in 0..1], "weight": 0..2, "reasoning": str }
    - Uses RBF/softmax interpolation across centers to produce smooth field.
    """
    import torch as _torch
    # Unwrap wrapper
    if isinstance(llm_dict_or_list, dict) and 'result' in llm_dict_or_list and isinstance(llm_dict_or_list['result'], list):
        points = llm_dict_or_list['result']
    else:
        points = llm_dict_or_list

    if not isinstance(points, list) or len(points) == 0:
        return _torch.ones((H_feat, W_feat), device=device)

    # Collect centers and weights
    centers = []
    weights = []
    for it in points:
        if not isinstance(it, dict):
            continue
        c = it.get('center'); w = it.get('weight')
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            continue
        try:
            x = float(c[0]); y = float(c[1]); wv = float(w)
        except Exception:
            continue
        # Clamp inputs
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        wv = 0.0 if wv < 0.0 else (2.0 if wv > 2.0 else wv)
        centers.append([x, y]); weights.append(wv)

    if len(centers) == 0:
        return _torch.ones((H_feat, W_feat), device=device)

    centers_t = _torch.tensor(centers, dtype=_torch.float32, device=device)  # [N,2]
    weights_t = _torch.tensor(weights, dtype=_torch.float32, device=device)  # [N]

    # Build normalized coordinate grid [0,1]x[0,1] with y increasing downward to match feature indexing
    ys = _torch.linspace(0.0, 1.0, steps=H_feat, device=device)
    xs = _torch.linspace(0.0, 1.0, steps=W_feat, device=device)
    yy, xx = _torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    grid = _torch.stack([xx, yy], dim=-1)  # [H,W,2]

    # Compute squared distances to each center: [H,W,N]
    diff = grid.unsqueeze(-2) - centers_t.unsqueeze(0).unsqueeze(0)  # [H,W,1,2] - [1,1,N,2]
    d2 = (diff ** 2).sum(dim=-1)  # [H,W,N]

    # Softmax over negative squared distances (RBF) to interpolate
    denom = 2.0 * (sigma ** 2)
    logits = -d2 / (denom + 1e-12)  # [H,W,N]
    weights_norm = _torch.softmax(logits, dim=-1)  # [H,W,N]
    att = (weights_norm * weights_t.view(1, 1, -1)).sum(dim=-1)  # [H,W]

    # Clamp final map to [0,2]
    att = att.clamp(0.0, 2.0)
    return att

# def llm_coord_to_attention_map_back(llm_dict_or_list, H_feat: int, W_feat: int, device, sigma: float = 0.15):
#     """
#     Convert coord-based LLM list to a [H_feat, W_feat] attention map tensor using smooth interpolation.
#     - Input accepts either wrapper { 'result': [...] } or the list directly.
#     - Each item: { "center": [x in 0..1, y in 0..1], "weight": 0..2, "reasoning": str }
#     - Uses RBF/softmax interpolation across centers to produce smooth field.
#     """
#     import torch as _torch
#     # Unwrap wrapper
#     if isinstance(llm_dict_or_list, dict) and 'result' in llm_dict_or_list and isinstance(llm_dict_or_list['result'], list):
#         points = llm_dict_or_list['result']
#     else:
#         points = llm_dict_or_list

#     if not isinstance(points, list) or len(points) == 0:
#         return _torch.ones((H_feat, W_feat), device=device)

#     # Collect centers and weights
#     centers = []
#     weights = []
#     for it in points:
#         if not isinstance(it, dict):
#             continue
#         c = it.get('center'); w = it.get('weight')
#         if not isinstance(c, (list, tuple)) or len(c) != 2:
#             continue
#         try:
#             x = float(c[0]); y = float(c[1]); wv = float(w)
#         except Exception:
#             continue
#         # Clamp inputs
#         x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
#         y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
#         wv = 0.0 if wv < 0.0 else (2.0 if wv > 2.0 else wv)
#         centers.append([x, y]); weights.append(wv)

#     if len(centers) == 0:
#         return _torch.ones((H_feat, W_feat), device=device)

#     centers_t = _torch.tensor(centers, dtype=_torch.float32, device=device)  # [N,2]
#     weights_t = _torch.tensor(weights, dtype=_torch.float32, device=device)  # [N]

#     # Build normalized coordinate grid [0,1]x[0,1] with y increasing downward to match feature indexing
#     ys = _torch.linspace(0.0, 1.0, steps=H_feat, device=device)
#     xs = _torch.linspace(0.0, 1.0, steps=W_feat, device=device)
#     yy, xx = _torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
#     grid = _torch.stack([xx, yy], dim=-1)  # [H,W,2]

#     # Compute squared distances to each center: [H,W,N]
#     diff = grid.unsqueeze(-2) - centers_t.unsqueeze(0).unsqueeze(0)  # [H,W,1,2] - [1,1,N,2]
#     d2 = (diff ** 2).sum(dim=-1)  # [H,W,N]

#     # Softmax over negative squared distances (RBF) to interpolate
#     denom = 2.0 * (sigma ** 2)
#     logits = -d2 / (denom + 1e-12)  # [H,W,N]
#     weights_norm = _torch.softmax(logits, dim=-1)  # [H,W,N]
#     att = (weights_norm * weights_t.view(1, 1, -1)).sum(dim=-1)  # [H,W]

#     # Clamp final map to [0,2]
#     att = att.clamp(0.0, 2.0)
#     return att


# def load_dino_map(dino_map_file,image_basename):
#     """
#     Load DINO map from image path.
#     Sample key in dino_map_dict:
#     {
#         "37.748316719_-122.410546068@2014-08_cleaned_2.jpg": {
#             "node_id": [],
#             "phrase": [],
#             "annotation_bbox": [],
#             "confidence": []
#             }
#         }
#     }
#     """
#     dino_map_dict = safe_json_load(dino_map_file)
#     dino_map = dino_map_dict[image_basename]

#     return dino_map

if __name__ == "__main__":
    # print(dash_coordinates("ABERDEEN TUNNEL_33%22.25640850710279_114.1799050738104@2023-07@60@210.0@-12.0@2048x4096.jpg"))
    # print(extract_place_id("__x9FdRdkfGBozPCf9C-ag%35.653654_139.692150@NA@NA@NA@381610.02@3946322.40@000.png", "dash"))

    query_folder = "F:/VPR_dataset/hk-urban/hk_flood"
    target_folder = "F:/VPR_dataset/hk-urban/gsv_pano_extracted"
    query_path = os.path.join(query_folder, "Nathan Rd 623&Shantung St%22.317515_114.169682_1.jpg")
    target_path = os.path.join(target_folder, "@0551294.51@4177187.54@10@S@037.74054@-122.41781@6ndRH0-kf_BJq0ZASDu5ZQ@@300@@@@201311@@.jpg")
    print(get_coordinates_from_path("dash", query_path))
    print(get_coordinates_from_path("parse", target_path))