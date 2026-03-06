# from schema_config import SchemaConfig
import json
import os
import logging
import re

logger = logging.getLogger(__name__)


class JsonDataCleaner:
    """
    Modular class for cleaning and verifying JSON data for graph-based tasks.
    Provides batch and single-file cleaning, LLM-based cleaning, and structure/content verification.
    """
    # def __init__(self):
    #     self.node_schemas = SchemaConfig.get_node_schemas()
    #     self.edge_schemas = SchemaConfig.get_edge_schemas()
    #     self.weight_schemas = SchemaConfig.get_weight_schemas()

    # 1. Manual string cleaning --------------------------------
    def clean_initial(self, json_content):
        if not isinstance(json_content, str):
            try:
                json_content = json.dumps(json_content)
            except Exception:
                return ""
        cleaned = json_content.strip()
        # Early handle explicit None control signal (from prompts)
        if cleaned.strip().lower() == 'none':
            return 'None'
        cleaned = re.sub(r'```json', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)
        # Remove trailing commas before closing brackets/braces
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        # Prefer extracting a JSON array if present; otherwise extract an object
        array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        obj_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if array_match and obj_match:
            # Choose whichever appears first in the text
            cleaned = array_match.group(0) if array_match.start() < obj_match.start() else obj_match.group(0)
        elif array_match:
            cleaned = array_match.group(0)
        elif obj_match:
            cleaned = obj_match.group(0)
        cleaned = cleaned.strip()
        return cleaned

    # 2. LLM-based cleaning (bot must have .chat method) --------------------------------
    def clean_llm(self, json_content, bot):
        try:
            prompt = f"Please fix the following JSON so it is valid and follows the expected structure.\n{json_content}"
            response = bot.chat(text_prompt=prompt)
            cleaned = self.clean_initial(response)
            try:
                return json.loads(cleaned)
            except Exception:
                return None
        except Exception as e:
            logger.error(f"LLM cleaning failed: {e}")
            return None

    # PIPELINE A: SviAgent, FOR PROMPTs: svi_att, 1_round,grid_4 ==============================================
    # 3. Format verification: is it a dict, does it parse --------------------------------
    def verify_json_format(self, json_data):
        return isinstance(json_data, dict)

    # 4. Content structure: has necessary keys and values --------------------------------
    def verify_att_content(self, json_data, grid_num = 4):
        """
        Verify if JSON data has valid content structure for given grid, check
        - all necessary keys are present: 
            for grid_num = 3, it's {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"}
            for grid_num = 4, it's {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4"}
        - all values are from 0 to 2
        Args:
            json_data: JSON data {"A1": 1, "A2": 2, "A3": 0.5, ...}, value for each key should be from 0 to 2
            grid_num: number of grids, default is 3, for 3x3 grid
        Returns:
            True if valid, False otherwise
        """

        if not isinstance(json_data, dict):
            return False

        required_keys = {f"{chr(ord('A') + r)}{c}" for r in range(grid_num) for c in range(1, grid_num + 1)}
        if not required_keys.issubset(json_data.keys()):
            return False

        for key in required_keys:
            value = json_data.get(key)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return False
            if not (0.0 <= numeric_value <= 2.0):
                return False

        return True

    def process_response_json(self, response, grid_num = 4):
        """
        Clean and verify a single response string. Returns dict with:
        - cleaned_json: the cleaned/parsed JSON (or None)
        - json_format_verified: bool
        - json_content_verified: bool
        - no_weighting: bool (True if LLM signaled 'None')
        """
        # 1. Try initial clean/parse
        cleaned = self.clean_initial(response)
        # Handle explicit control signal "None" (case-insensitive)
        if isinstance(cleaned, str) and cleaned.strip().lower() == 'none':
            return {
                'cleaned_json': None,
                'json_format_verified': True,
                'json_content_verified': True,
                'no_weighting': True
            }
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
            
        # 2. Verify
        json_format_verified = self.verify_json_format(parsed) if parsed is not None else False
        json_content_verified = self.verify_att_content(parsed) if parsed is not None else False
        
        return {
            'cleaned_json': parsed,
            'json_format_verified': json_format_verified,
            'json_content_verified': json_content_verified,
            'no_weighting': False
        }

    # PIPELINE B: SviAgent, check content structure ==============================================
    def process_response_list(self, response):
        """
        Response is a list of dicts, sample:
            [
                {
                    "center": [x_coord, y_coord],
                    "weight": weight_value,
                    "reasoning": "brief_description"
                }
            ]
        Returns:
            dict with:
            - cleaned_json: the cleaned/parsed JSON (or None)
            - json_format_verified: bool
            - json_content_verified: bool
            - no_weighting: bool (True if LLM signaled 'None')
        """
        cleaned = self.clean_initial(response)
        # Handle explicit control signal "None"
        if isinstance(cleaned, str) and cleaned.strip().lower() == 'none':
            return {
                'cleaned_json': None,
                'json_format_verified': True,
                'json_content_verified': True,
                'no_weighting': True
            }
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
            
        json_format_verified = self.verify_list_format(parsed) if parsed is not None else False
        json_content_verified = self.verify_list_content(parsed) if parsed is not None else False
        return {
            'cleaned_json': parsed,
            'json_format_verified': json_format_verified,
            'json_content_verified': json_content_verified,
            'no_weighting': False
        }

    def verify_list_format(self, json_data):
        """
        Verify if JSON data is a list format.
        Args:
            json_data: parsed JSON data
        Returns:
            True if valid list format, False otherwise
        """
        return isinstance(json_data, list)

    def verify_list_content(self, json_data):
        """
        Verify if JSON data has valid content structure for list items.
        Each item should have "center", "weight", and "reasoning" fields.
        Args:
            json_data: parsed JSON data (should be a list)
        Returns:
            True if valid content structure, False otherwise
        """
        if not isinstance(json_data, list):
            return False
        
        # Check if list is empty
        if len(json_data) == 0:
            return False
        
        required_fields = {"center", "weight", "reasoning"}
        
        for item in json_data:
            if not isinstance(item, dict):
                return False
            
            # Check if all required fields are present
            if not required_fields.issubset(item.keys()):
                return False
            
            # Validate center field: should be a list with 2 numeric coordinates
            center = item.get("center")
            if not isinstance(center, list) or len(center) != 2:
                return False
            
            try:
                x_coord, y_coord = float(center[0]), float(center[1])
                # Check if coordinates are within valid range [0.0, 1.0]
                if not (0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0):
                    return False
            except (TypeError, ValueError, IndexError):
                return False
            
            # Validate weight field: should be a numeric value between 0.0 and 2.0
            weight = item.get("weight")
            try:
                weight_value = float(weight)
                if not (0.0 <= weight_value <= 2.0):
                    return False
            except (TypeError, ValueError):
                return False
            
            # Validate reasoning field: should be a non-empty string
            reasoning = item.get("reasoning")
            if not isinstance(reasoning, str) or len(reasoning.strip()) == 0:
                return False
        
        return True


    # 5. Patch content: patch the JSON data to have all necessary keys and values =====================================
    def patch_att_content(self, json_data, grid_num = 4):
        """
        Patch the JSON data to have all necessary keys and values.
        All missing keys will be set with default value 1.0, and all values above 2 will be set to 2.0, values below 0 will be set to 0.0.
        Args:
            json_data: JSON data {"A1": 1, "A2": 2, "A3": 0.5, ...}
            grid_num: number of grids, default is 3, for 3x3 grid
        Returns:
            JSON data with all necessary keys and fixed values
        """

        if not isinstance(json_data, dict):
            json_data = {}

        required_keys = {f"{chr(ord('A') + r)}{c}" for r in range(grid_num) for c in range(1, grid_num + 1)}

        def clamp_to_unit_interval(value):
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = 1.0
            if v < 0.0:
                return 0.0
            if v > 2.0:
                return 2.0
            return v

        patched = dict(json_data)
        for key in required_keys:
            if key in patched:
                patched[key] = clamp_to_unit_interval(patched[key])
            else:
                patched[key] = 1.0

        return patched

    def patch_list_content(self, json_list):
        """
        Patch axis-focus list content to a valid structure.
        - Keep only dict items containing required fields
        - Coerce center to two floats within [0.0, 1.0]
        - Clamp weight to [0.0, 2.0]
        - Ensure reasoning is a non-empty string
        """
        if not isinstance(json_list, list):
            return []

        patched_items = []
        for item in json_list:
            if not isinstance(item, dict):
                continue
            center = item.get("center")
            weight = item.get("weight")
            reasoning = item.get("reasoning")

            # Validate and coerce center
            if isinstance(center, (list, tuple)) and len(center) == 2:
                try:
                    x = float(center[0])
                    y = float(center[1])
                except (TypeError, ValueError):
                    continue
                # Clamp to [0,1]
                x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
                center_fixed = [x, y]
            else:
                continue

            # Validate and coerce weight
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            w = 0.0 if w < 0.0 else (2.0 if w > 2.0 else w)

            # Validate reasoning
            if not isinstance(reasoning, str):
                continue
            reasoning_fixed = reasoning.strip()
            if len(reasoning_fixed) == 0:
                continue

            patched_items.append({
                "center": center_fixed,
                "weight": w,
                "reasoning": reasoning_fixed
            })

        return patched_items

    def clean_att_content_batch_pipeline(self, json_save_folder):
        """
        Validate and patch saved JSONs in-place.
        - Supports both grid dict format and axis-focus list format.
        - Writes patched content back to the same file.
        - Avoids deleting files while they are open.
        """
        format_false = 0
        content_false = 0
        total_count = 0
        for file in os.listdir(json_save_folder):
            if file.endswith(".json"):
                total_count += 1
                json_path = os.path.join(json_save_folder, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except Exception:
                    format_false += 1
                    continue

                # Remove optional top-level timestamp field if present
                json_data.pop('timestamp', None)

                # Expect a wrapper dict with keys including result/json_*_verified
                result = json_data.get('result', None)
                # Handle explicit no_weighting control: treat as valid with None result
                if json_data.get('no_weighting', False) is True and result is None:
                    json_data['json_format_verified'] = True
                    json_data['json_content_verified'] = True
                    try:
                        with open(json_path, 'w', encoding='utf-8') as wf:
                            json.dump(json_data, wf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    # Skip further checks for this file
                    continue
                if isinstance(result, str):
                    # Clean and parse stringified JSON result (handles ```json fences)
                    cleaned_result = self.clean_initial(result)
                    try:
                        result = json.loads(cleaned_result)
                        # Persist parsed structure back to json_data immediately
                        json_data['result'] = result
                    except Exception:
                        result = None
                if result is None:
                    format_false += 1
                    continue

                # Branch by result type
                if isinstance(result, dict):
                    # Grid attention case
                    is_valid = self.verify_att_content(result)
                    if not is_valid:
                        patched_data = self.patch_att_content(result)
                        is_valid = self.verify_att_content(patched_data)
                        json_data['result'] = patched_data
                    json_data['json_format_verified'] = True
                    json_data['json_content_verified'] = bool(is_valid)
                    if not is_valid:
                        content_false += 1
                    # Write back
                    try:
                        with open(json_path, 'w', encoding='utf-8') as wf:
                            json.dump(json_data, wf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                elif isinstance(result, list):
                    # Axis-focus list case
                    is_valid = self.verify_list_content(result)
                    if not is_valid:
                        patched_list = self.patch_list_content(result)
                        is_valid = self.verify_list_content(patched_list)
                        json_data['result'] = patched_list
                    json_data['json_format_verified'] = True
                    json_data['json_content_verified'] = bool(is_valid)
                    if not is_valid:
                        content_false += 1
                    # Write back
                    try:
                        with open(json_path, 'w', encoding='utf-8') as wf:
                            json.dump(json_data, wf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                else:
                    # Unknown result type
                    format_false += 1
                    continue

        print(f"Cleaned {total_count} JSON files, {format_false} format false, {content_false} content false")

    # 6. Collect failed files and remove them ==============================================
    def remove_failed_file(self, json_save_folder):
        """
        Remove failed files in the json_save_folder.
        """
        failed_file_list = self.collect_failed_file(json_save_folder)
        execute_boolean = input(f"Found {len(failed_file_list)} failed files, do you want to remove them? (y/n): ")
        if execute_boolean == "y":
            for file in failed_file_list:
                os.remove(file)
            print(f"Removed {len(failed_file_list)} failed files")
        else:
            print(f"Found {len(failed_file_list)} failed files, not removed")

    def collect_failed_file(self, json_save_folder):
        """
        Collect failed files in the json_save_folder.
        """
        failed_file_list = []
        for file in os.listdir(json_save_folder):
            if file.endswith(".json"):
                json_path = os.path.join(json_save_folder, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        json_format_verified = json_data.get('json_format_verified', False)
                        json_content_verified = json_data.get('json_content_verified', False)
                        if not json_format_verified or not json_content_verified:
                            failed_file_list.append(json_path)
                            print(f"Found failed file: {file}")
                except Exception:
                    failed_file_list.append(json_path)
        # print(f"Found {len(failed_file_list)} failed files")
        return failed_file_list

    def collect_failed_file(self, json_save_folder):
        """
        Collect failed files in the json_save_folder.
        """
        failed_file_list = []
        for file in os.listdir(json_save_folder):
            if file.endswith(".json"):
                json_path = os.path.join(json_save_folder, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        json_format_verified = json_data.get('json_format_verified', False)
                        json_content_verified = json_data.get('json_content_verified', False)
                        if not json_format_verified or not json_content_verified:
                            failed_file_list.append(json_path)
                except Exception:
                    failed_file_list.append(json_path)
        print(f"Found {len(failed_file_list)} failed files")
        return failed_file_list

# =============== PIPELINE: LLM FOR RERANKING TOP 100 ==============================================
    def process_top100_json(self, response, bot=None, llm_clean=False):
        """
        Clean and verify a single response string. Returns dict with:
        - cleaned_json: the cleaned/parsed JSON (or None)
        - json_format_verified: bool
        - json_content_verified: bool
        - original_response: the original string
        """
        # 1. Try initial clean/parse
        cleaned = self.clean_initial(response)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
            
        # 2. If failed and llm_clean, try LLM
        if parsed is None and llm_clean and bot is not None:
            parsed = self.clean_llm(cleaned, bot)
            # always maually clean llm's response first because it may contain ```json and ```
            parsed = self.clean_initial(parsed)
            
        # 3. Verify
        json_format_verified = self.verify_json_format(parsed) if parsed is not None else False
        json_content_verified = self.verify_json_top100_content_structure(parsed) if parsed is not None else False

        if parsed is not None:       
            return {
                'cleaned_json': parsed,
                'json_format_verified': json_format_verified,
                'json_content_verified': json_content_verified,
                'original_response': None
            }
        else: # only save original response if failed to parse
            return {
                'cleaned_json': None,
                'json_format_verified': json_format_verified,
                'json_content_verified': json_content_verified,
                'original_response': response
            }            
    
    def verify_json_top100_content_structure(self, json_data):
        """
        Verify if JSON data has valid dictionary structure for list items in "matches" key.
        Returns True if valid, False otherwise.
        """
        if 'top_100' not in json_data:
            return False
        matches = json_data['matches']
        if not isinstance(matches, list):
            return False
        for match in matches:
            if not isinstance(match, dict):
                return False
            if "image_number" not in match:
                return False
            if "confidence" not in match:
                return False
        return True

if __name__ == "__main__":
    # Configuration
    svi_folder = "C:/VPR_temp/hk-urban/hk_flood"
    json_folder = os.path.join(svi_folder, "att_axis@gemini-2.5-flash")

    # --- Clean JSONs before using GraphMatcher ---
    cleaner = JsonDataCleaner()
    # print(f"Cleaning query folder: {query_json_folder}")
    # cleaner.clean_json_batch_pipeline(query_json_folder, force_clean=True)
    print(f"Cleaning target folder: {json_folder}")
    # cleaner.clean_att_content_batch_pipeline(database_json_folder)
    cleaner.remove_failed_file(json_folder)