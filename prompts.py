"""
RECORD OF PROMPTS
The generation of node description is relative stable. Appeding distinctive features can improve difference beteween similar nodes.
The side understanding is complex and not stable.
    Original prompt:
    left/right (use left/right when intersection visible - standing at center facing towards street segment)
    camera_left/camera_right (use camera_left/camera_right when no intersection - from viewing perspective)

"""

class PromptManager:
    """Elegant prompt management with categorization and easy access."""
    
    def __init__(self):
        self._prompts = {
            "image_1": {
                # work good for urban view, but hullucinated for natural view
                "same_view_v1": """
                create the flooding scene with changed weather condition of this place, keeping same view point, and ensure consistent appearances for the building, and basic landscape for this place, as well as their relative spatial position based on given image.
                you may change the cars, people, and other temporary objects, but keep the permanent infrastructures and their relative spatial position based on given image.
                Keep the realistic style of the generated image referring to the given image.
                """,
                # work for natural view (good)
                "same_view_v2": """
                Simulate a realistic flood at this location, maintaining the original camera viewpoint. Preserve the primary subjects and the overall composition of the scene, including topography, major vegetation, and any structures. These elements should retain their relative positions and core appearance. Introduce realistic floodwaters and render the scene with overcast, stormy weather. The final image must match the photorealistic style of the original.
                """,
                # more conservative
                "same_view_v2.1": """
                Simulate a realistic slight inundation at this location, maintaining the original camera viewpoint. Preserve the primary subjects and the overall composition of the scene, including topography, major vegetation, and any structures. These elements should retain their relative positions and core appearance. Introduce realistic floodwaters and render the scene with overcast, stormy weather. The final image must match the photorealistic style of the original.
                """,
                "same_view_v2.2": """
                Simulate a realistic slight inundation at this location. Strictly preserve all original buildings and infrastructure. Maintain the original camera viewpoint. Add realistic floodwaters and overcast, stormy weather. The final image should look like a photo taken in the rain, with imperfections like water droplets or smudges on the lens.
                """,
                "same_view_v2.3": """
                Simulate a realistic slight inundation at this location. Strictly preserve all original buildings and infrastructure. Maintain the original camera viewpoint. Add realistic floodwaters and overcast, stormy weather. The final image should look like a photo taken in the rain, with imperfections like a few water droplets and a smear across a portion of the lens.
                """,
                "same_view_v2.4": """
                Simulate a realistic slight inundation at this location. Strictly preserve all original buildings and infrastructure. Maintain the original camera viewpoint. Add realistic floodwaters and overcast, stormy weather. Add imperfections like water streaks and smudges on the lens, as if it were wiped hastily.
                """,

                # Introduce potential damage (might cause hullucination)
                "same_view_v3": """
                Simulate a realistic flood at this location, maintaining the original camera viewpoint. Preserve the primary subjects and the overall composition of the scene, including topography, major vegetation, and any structures. These elements should retain their relative positions and core appearance. 
                Introduce turbulent, debris-laden floodwaters that realistically interact with the environment, showing signs of displaced vegetation and accumulated floating material. Render the scene with overcast, stormy weather. The final image must match the photorealistic style of the original.
                """,
                "different_view": """
                Create the flooding scene for this street view image, with changed weather condition, using slightly changed viewpoint so it looks like photos from social media post. Ensure consistent appearances for the building and other infrastructures for this place, as well as their relative spatial position based on given image
                """
            },

            # Survey and Analysis Prompts
            # Survey 1: node description first, edge description second
            "svi_att": {
                "1_round": """
                    # VPR Geo-localization: Spatial Uniqueness Weighting

                    **TASK:** Weight each 3x3 grid cell (0-2 scale) based on spatial uniqueness within Tokyo for precise location matching.

                    **GRID:** 
                    A1: top-left, A2: top-center, A3: top-right
                    B1: middle-left, B2: center, B3: middle-right
                    C1: bottom-left, C2: bottom-center, C3: bottom-right

                    **WEIGHTING LOGIC:**

                    **HIGH WEIGHTS (1.6-2.0):** Spatially unique in Tokyo
                    - Rare architectural details, unusual building features
                    - Distinctive facades, unique structural elements
                    - Sharp, detailed building textures/patterns
                    - Uncommon urban installations or specialized infrastructure

                    **MEDIUM WEIGHTS (1.0-1.5):** Moderately distinctive
                    - Well-detailed standard buildings with clear architectural features
                    - Street-level infrastructure with visible specificity
                    - Clear signage, distinctive storefronts

                    **LOW WEIGHTS (0.3-0.9):** Common Tokyo elements
                    - Typical building styles (abundant citywide)
                    - Generic street infrastructure, standard traffic elements

                    **MINIMAL WEIGHTS (0.0-0.2):** Non-localizable
                    - Sky, vegetation, generic pavement
                    - People, vehicles, temporary objects
                    - Blurred or indistinct regions

                    **CRITICAL PRINCIPLE:** Prioritize what makes THIS specific spot unique within Tokyo.

                    **OUTPUT FORMAT:**
                    {"A1":0.0,"A2":0.0,"A3":0.0,"B1":0.0,"B2":0.0,"B3":0.0,"C1":0.0,"C2":0.0,"C3":0.0}
                """,
                "grid_4": """
                    # VPR Geo-localization: Scene Composition Weighting

                    **TASK:** 
                    Your goal is to determine if an image requires spatial weighting for Visual Place Recognition (VPR) and, if so, to provide those weights. Follow this two-step decision process strictly.

                    STEP 1: DECIDE THE SCENE TYPE
                    First, classify the image into one of two categories. This is the most critical step.

                    A) 'Single Landmark' Image: The image is dominated by one primary, cohesive structure.

                    Criteria: This includes focused shots of a single building's facade, a prominent monument, a statue, or a bridge, etc. The object acts as a single, unambiguous identifier.
                    Crucial Rule: Even if this single landmark has unique details (e.g., intricate windows, a brand name, specific textures), it must be treated as one whole entity. Do not segment a single building into parts. Applying different weights to parts of one building is incorrect and adds noise.
                    ACTION: If the image is a 'Single Landmark', STOP. Your entire and only output must be the single string: "None"

                    B) 'Complex Scene' Image: The image contains multiple, distinct, and geographically significant objects that contribute differently to localization.

                    Criteria: This includes street corners with several buildings, plazas with fountains and surrounding structures, or street views with varied storefronts and unique street furniture. The combination of these separate objects defines the location.
                    ACTION: If the image is a 'Complex Scene', PROCEED to Step 2 to generate weights.

                    STEP 2: GENERATE WEIGHTS (FOR 'COMPLEX SCENE' IMAGES ONLY)
                    If and only if you identified the image as a 'Complex Scene' in Step 1, assign weights to each 4x4 grid cell (0-2 scale) based on their uniqueness for localization in San Francisco.

                    **GRID:** 
                    A1 A2 A3 A4
                    B1 B2 B3 B4
                    C1 C2 C3 C4
                    D1 D2 D3 D4

                    **WEIGHTING LOGIC:**

                    **HIGH WEIGHTS (1.6-2.0):** Spatially unique in San Francisco
                    - Rare architectural details, unusual building features
                    - Distinctive facades, unique structural elements
                    - Sharp, detailed building textures/patterns
                    - Uncommon urban installations or specialized infrastructure

                    **MEDIUM WEIGHTS (1.0-1.5):** Moderately distinctive
                    - Well-detailed standard buildings with clear architectural features
                    - Street-level infrastructure with visible specificity
                    - Clear signage, distinctive storefronts

                    **LOW WEIGHTS (0.3-0.9):** Common San Francisco elements
                    - Typical building styles (abundant citywide)
                    - Generic street infrastructure, standard traffic elements

                    **MINIMAL WEIGHTS (0.0-0.2):** Non-localizable
                    - Sky, vegetation, generic pavement
                    - People, vehicles, temporary objects
                    - Blurred or indistinct regions

                    **CRITICAL PRINCIPLE:** Prioritize what makes THIS specific spot unique within San Francisco.

                    **OUTPUT FORMAT:**                  
                    {"A1":0.0,"A2":0.0,"A3":0.0,"A4":0.0,"B1":0.0,"B2":0.0,"B3":0.0,"B4":0.0,"C1":0.0,"C2":0.0,"C3":0.0,"C4":0.0,"D1":0.0,"D2":0.0,"D3":0.0,"D4":0.0}
                """,

                "grid_4_nonone": """
                    # VPR Geo-localization: Spatial Uniqueness Weighting

                    **TASK:** Weight each 4x4 grid cell (0-2 scale) based on spatial uniqueness within San Francisco for precise location matching.

                    **GRID:** 
                    A1 A2 A3 A4
                    B1 B2 B3 B4
                    C1 C2 C3 C4
                    D1 D2 D3 D4

                    **WEIGHTING LOGIC:**

                    **HIGH WEIGHTS (1.6-2.0):** Spatially unique in San Francisco
                    - Rare architectural details, unusual building features
                    - Distinctive facades, unique structural elements
                    - Sharp, detailed building textures/patterns
                    - Uncommon urban installations or specialized infrastructure

                    **MEDIUM WEIGHTS (1.0-1.5):** Moderately distinctive
                    - Well-detailed standard buildings with clear architectural features
                    - Street-level infrastructure with visible specificity
                    - Clear signage, distinctive storefronts

                    **LOW WEIGHTS (0.3-0.9):** Common San Francisco elements
                    - Typical building styles (abundant citywide)
                    - Generic street infrastructure, standard traffic elements

                    **MINIMAL WEIGHTS (0.0-0.2):** Non-localizable
                    - Sky, vegetation, generic pavement
                    - People, vehicles, temporary objects
                    - Blurred or indistinct regions

                    **CRITICAL PRINCIPLE:** Prioritize what makes THIS specific spot unique within San Francisco.

                    **OUTPUT FORMAT:**
                    First, assess if the image contains multiple distinct landmarks or is dominated by a single one.
                    If the image is dominated by a single unique building or object (e.g., a close-up of one building, a statue, etc.), output the single string: None
                    If the image is a complex scene with multiple objects that contribute differently to localization, provide a JSON content as follows:                    
                    {"A1":0.0,"A2":0.0,"A3":0.0,"A4":0.0,"B1":0.0,"B2":0.0,"B3":0.0,"B4":0.0,"C1":0.0,"C2":0.0,"C3":0.0,"C4":0.0,"D1":0.0,"D2":0.0,"D3":0.0,"D4":0.0}
                """,

                "axis_focus": """
                    # VPR Geo-localization: Scene Composition Weighting

                    **TASK:** 
                    Your goal is to determine if an image requires spatial weighting for Visual Place Recognition (VPR) and, if so, to provide those weights. Follow this two-step decision process strictly.

                    STEP 1: DECIDE THE SCENE TYPE
                    First, classify the image into one of two categories. This is the most critical step.

                    A) 'Single Landmark' Image: The image is dominated by one primary, cohesive structure.

                    Criteria: This includes focused shots of a single building's facade, a prominent monument, a statue, or a bridge, etc. The object acts as a single, unambiguous identifier.
                    Crucial Rule: Even if this single landmark has unique details (e.g., intricate windows, a brand name, specific textures), it must be treated as one whole entity. Do not segment a single building into parts. Applying different weights to parts of one building is incorrect and adds noise.
                    ACTION: If the image is a 'Single Landmark', STOP. Your entire and only output must be the single string: "None"

                    B) 'Complex Scene' Image: The image contains multiple, distinct, and geographically significant objects that contribute differently to localization.

                    Criteria: This includes street corners with several buildings, plazas with fountains and surrounding structures, or street views with varied storefronts and unique street furniture. The combination of these separate objects defines the location.
                    ACTION: If the image is a 'Complex Scene', PROCEED to Step 2 to generate weights.

                    STEP 2: GENERATE WEIGHTS (FOR 'COMPLEX SCENE' IMAGES ONLY)
                    If and only if you identified the image as a 'Complex Scene' in Step 1, identify 3-8 key regions and assign weights based on their uniqueness for localization in the city.
                    The city is {city}, and the spatial uniqueness is the uniqueness of the region in the whole city.

                    **COORDINATE SYSTEM:**

                    Image coordinates range from (0.0, 0.0) at top-left to (1.0, 1.0) at bottom-right
                    Center coordinates represent the focal point of each distinctive region

                    **WEIGHTING LOGIC:**

                    **HIGH WEIGHTS (1.6-2.0):** Spatially unique in {city}
                    - Rare architectural details, unusual building features
                    - Distinctive facades, unique structural elements
                    - Sharp, detailed building textures/patterns
                    - Uncommon urban installations or specialized infrastructure

                    **MEDIUM WEIGHTS (1.0-1.5):** Moderately distinctive in {city}
                    - Well-detailed standard buildings with clear architectural features
                    - Street-level infrastructure with visible specificity
                    - Clear signage, distinctive storefronts

                    **LOW WEIGHTS (0.3-0.9):** Common elements in {city}
                    - Typical building styles (abundant citywide)
                    - Generic street infrastructure, standard traffic elements

                    **MINIMAL WEIGHTS (0.0-0.2):** Non-localizable in {city}
                    - Sky, vegetation, generic pavement
                    - People, vehicles, temporary objects
                    - Blurred or indistinct regions

                    **SELECTION STRATEGY:**

                    Focus on 3-8 most important regions that deviate significantly from default weight (1.0)
                    Prioritize regions with highest uniqueness (>1.5) and lowest uniqueness (<0.5)
                    Skip regions that would receive approximately default weight (0.9-1.1)

                    **OUTPUT FORMAT:**
                    [
                        {
                            "center": [x_coord, y_coord],
                            "weight": weight_value,
                            "reasoning": "brief_description"
                        }
                    ]
                """,

                "axis_focus_nonone": """
                    # VPR Geo-localization: Spatial Uniqueness Weighting

                    **TASK:** Identify distinctive regions in the street view image and assign weights based on spatial uniqueness within San Francisco for precise location matching.

                    **COORDINATE SYSTEM:**

                    Image coordinates range from (0.0, 0.0) at top-left to (1.0, 1.0) at bottom-right
                    Center coordinates represent the focal point of each distinctive region

                    **WEIGHTING LOGIC:**

                    **HIGH WEIGHTS (1.6-2.0):** Spatially unique in San Francisco
                    - Rare architectural details, unusual building features
                    - Distinctive facades, unique structural elements
                    - Sharp, detailed building textures/patterns
                    - Uncommon urban installations or specialized infrastructure

                    **MEDIUM WEIGHTS (1.0-1.5):** Moderately distinctive
                    - Well-detailed standard buildings with clear architectural features
                    - Street-level infrastructure with visible specificity
                    - Clear signage, distinctive storefronts

                    **LOW WEIGHTS (0.3-0.9):** Common San Francisco elements
                    - Typical building styles (abundant citywide)
                    - Generic street infrastructure, standard traffic elements

                    **MINIMAL WEIGHTS (0.0-0.2):** Non-localizable
                    - Sky, vegetation, generic pavement
                    - People, vehicles, temporary objects
                    - Blurred or indistinct regions

                    **SELECTION STRATEGY:**

                    Focus on 3-8 most important regions that deviate significantly from default weight (1.0)
                    Prioritize regions with highest uniqueness (>1.5) and lowest uniqueness (<0.5)
                    Skip regions that would receive approximately default weight (0.9-1.1)

                    **OUTPUT FORMAT:**
                    [
                        {
                            "center": [x_coord, y_coord],
                            "weight": weight_value,
                            "reasoning": "brief_description"
                        }
                    ]
                """,

                "axis_focus_minimal": """
                    Analyze the image for VPR spatial weighting in {city}.

                    Identify 3-8 key regions and assign weights based on uniqueness for localization in {city}:
                    - 1.6–2.0: Spatially unique
                    - 1.0–1.5: Moderately distinctive
                    - 0.3–0.9: Common
                    - 0.0–0.2: Non-localizable (sky, vegetation, people, vehicles)

                    Skip regions near weight 1.0. Use coordinates (0.0, 0.0) top-left to (1.0, 1.0) bottom-right.

                    [
                        {
                            "center": [x_coord, y_coord],
                            "weight": weight_value,
                            "reasoning": "brief_description"
                        }
                    ]
                """
            },
            

            "survey_rerank": {
                "rerank_test": """
                **San Francisco Location Matching Task**

                Query Image: {IMAGE_1}
                Database Images: {IMAGE_2-IMAGE_11}

                Please identify which database image(s) show the same location as the query image in San Francisco. 

                Return:
                - Matched image numbers (eg. Image 2, Image 6) with confidence level (High/Medium/Low)
                - Brief reasoning for each match
                """,

                "rerank_1": """
                **San Francisco Location Matching Task**

                Query Image: {IMAGE_0} (Reference Image)
                Database Images: {IMAGE_1-100} (Candidate Images 1-100)

                Please identify which database image(s) show the same location as the query image in San Francisco.

                **Important:** When referencing database images in your response, use the numbers 1-100 corresponding to their display order (Image 1, Image 2, ..., Image 100).

                Return:
                - Top 5 matched image numbers from the database (1-100) with confidence level (High/Medium/Low)
                - Brief reasoning for each match
                - If no match is found, return null

                Expected output format:
                {
                    "matches": [
                        {"image_number": 13, "confidence": "High", "reasoning": "..."},
                        {"image_number": 47, "confidence": "Medium", "reasoning": "..."},
                        ...
                    ]
                }

                Expected output when no match is found:
                {
                    "matches": null
                }
                """
            },

            # Verification and Quality Check
            "verification": {
                "compare_svi_full": """
                A. TASK DESCRIPTION:
                The json content is generated by an urban surveyor for creating structured graph representations of street views for location identification.
                Compare the json with the street view image and following the CRITERIA, critically check:
                1. IF INTERSECTION VISIBLE:
                    - Street intersection (I) nodes are properly identified (if the corner is visible, there should be at least one intersection node, otherwise there should be no intersection node)
                    - All major visible street segments (S) are represented as separate segments nodes, and the number of street segment nodes is equal to the linked_segment_number of the intersection node
                2. IF INTERSECTION NOT VISIBLE:
                    - there should be no intersection node
                    - there should be only one street segment (S)node 
                3. All significant building (B) nodes are captured and not duplicated, with all attributes filled correctly or with "unknown" if not available from the image
                4. All land (L) nodes are captured and not duplicated, with all attributes filled correctly or with "unknown" if not available from the image
                5. All building (B) nodes and land (L) nodes are correctly connected using ALONG edges to ALL street segments (S) nodes it located on
                    - IF INTERSECTION VISIBLE and the building/land is at the corner, the B and L nodes have to connect to at least two street segments (S) nodes
                6. ALL building (B) nodes and land (L) nodes on same street segment side are connected using ADJACENT (touching boundaries between them) edges or SAMESIDE (no touching boundaries between them) edges.
                7. ALL building (B) nodes and land (L) nodes on same street segment but different side are connected using CROSSING edges.
                8. All street trees (T) nodes and facilities (F) nodes are correctly connected to the NEAR building (B) or land (L) nodes 
                9. Exported json format is flat (no nested objects) and no missing values

                B. REFERENCE SHEET:
                B-1. NODE ATTRIBUTES
                STREET SEGMENT (S) - Each physical street arm extending from an intersection, create separate segment nodes for each street direction
                id: S1, S2, S3...
                type: street segment
                street_name: visible signs or unknown
                lane_number: 1/2/3/4/5/>5/unknown (total vehicle lanes both directions, use car reference object to estimate)
                sidewalk_description: specific_description (width, material, etc.) or unknown

                BUILDING (B) - All structures facing streets
                id: B1, B2, B3...
                type: building
                building_type: residential/commercial/mixed/institutional/industrial/warehouse/office/bridge/other/unknown
                has_ground_retail: yes/no/unknown
                story_number: 1/2/3/4/5/>5/unknown
                width_description: specific_description or unknown
                color: white/gray/brown/red/blue/green/yellow/black/beige/tan/orange/unknown (list at most 2 major colors, eg. white/gray, red/blue, etc.)
                material: brick/concrete/wood/metal/glass/stone/stucco/vinyl/unknown
                distinctive_feature: specific_description or unknown

                LAND (L) - Open spaces/lots facing streets, street or road not included
                id: L1, L2, L3...
                type: land
                land_type: vacant/parking/park/construction/plaza/garden/sports/playground/unknown
                surface: grass/concrete/asphalt/dirt/gravel/sand/paved/unknown
                width_description: specific_description or unknown
                distinctive_feature: specific_description or unknown

                TREE (T) - Street trees along streets/sidewalks
                id: T1, T2, T3...
                type: tree
                size_description: specific_description or unknown
                distinctive_feature: specific_description or unknown

                FACILITY (F) - Street facilities along streets/sidewalks
                id: F1, F2, F3...
                type: facility
                facility_type: bus_stop/bench/lamp_post/traffic_light/sign/mailbox/fire_hydrant/parking_meter/trash_can/phone_booth/utility_pole/unknown 
                distinctive_feature: specific_description or unknown   

                B-2. EDGE TYPES & ATTRIBUTES:
                LINK (S-I): Street segments to intersections
                ALONG (B/L-S): Buildings/land to ALL facing street segments (Corner buildings/land: Connect to ALL street segments they face)
                ADJACENT (B-B/B-L/L-L): Neighboring buildings/land on same street segment side, touching boundaries between them
                - relative_position_description: specific_description (eg.flush/significant protrusion/etc.)
                SAMESIDE (B-B/B-L/L-L): Buildings/land on same street segment side, no touching boundaries between them
                - node_1, node_2, type
                CROSSING (B-B/B-L/L-L): Buildings/land on same street segment but on opposite street segment sides
                - offset: none/slight/major
                NEAR (T/F-B/L): Trees/facilities to same-side buildings/land
                - distance_meter: 1-20 (perpendicular distance from the street element to the nearest building/land boundary, using reference object dimensions to estimate)
                - relative_position_description: specific_description (eg.near the entrance/balcony/window/etc.)                

                B-3. EXPECTED JSON FORMAT:
                {"nodes":[{"id":"I1","type":"intersection","linked_segment_number":4}],"edges":[{"node_1":"B1","node_2":"S1","type":"ALONG"}]}
                Output only valid JSON in plain text - no explanatory text or code block.

                C. EVALUATION REQUIREMENTS:
                Based on the analysis of the json content, give a score for the surveyor's json content, and provide a specific description of the evaluation.
                If the json score is below 100, provide a specific suggestion for prompt refinement.

                EVALUATION OUTPUT FORMAT:
                {
                    "score": 0-100,
                    "evaluation": "specific description for the json content quality"
                    "refine_suggestion": "specific suggestion for prompt refinement"
                }

                Finally output the evaluation result, only valid JSON in plain text - no explanatory text or code block.
                """,

                "compare_svi_fixed": """
                TASK DESCRIPTION:
                The above json content is generated by a urban surveyor for depicting the spatial relationships for a the given street view images.
                The urban surveyor follows the SURVEY PROMPT to analyze the image and generates json content for creating structured graph representations of the street views.
                Compare the json with the street view image check according to the CRITERIA and REFERENCE SHEET:

                A. CRITERIA:
                1. IF INTERSECTION VISIBLE:
                    - Street intersection (I) nodes are properly identified (if the corner is visible, there should be at least one intersection node, otherwise there should be no intersection node);
                    - All major visible street segments (S) are represented as separate segments nodes, and the number of street segment nodes is equal to the linked_segment_number of the intersection node
                2. IF INTERSECTION NOT VISIBLE:
                    - there should be no intersection node;
                    - there should be only one street segment (S)node;
                3. All significant building (B) nodes are captured and not duplicated;
                4. All land (L) nodes are captured and not duplicated;
                5. All building (B) nodes and land (L) nodes are correctly connected using ALONG edges to ALL street segments (S) nodes it located on;
                    - IF INTERSECTION VISIBLE and the building/land is at the corner, the B and L nodes have to connect to at least two street segments (S) nodes;
                6. ALL building (B) nodes and land (L) nodes on same street segment side are connected using ADJACENT (touching boundaries between them) edges or SAMESIDE (no touching boundaries between them) edges;
                7. ALL building (B) nodes and land (L) nodes on same street segment but different side are connected using CROSSING edges;
                8. All street trees (T) nodes and facilities (F) nodes are correctly connected to the NEAR building (B) or land (L) nodes;
                9. ALL attributes for each node using exact values listed above are filled correctly and honestly;
                10. Exported json format is flat (no nested objects) and no missing values

                B. REFERENCE SHEET:
                B-1. EDGE TYPES & SEQUENTIAL CREATION ORDER
                LINK (S-I): Street segments to intersections
                - node_1, node_2, type

                ALONG (B/L-S): Buildings/land to ALL facing street segments (Corner buildings/land: Connect to ALL street segments they face)
                - node_1, node_2, type

                ADJACENT (B-B/B-L/L-L): Neighboring buildings/land on same street segment side, touching boundaries between them
                - node_1, node_2, type, relative_position_description: specific_description (eg.flush/significant protrusion/etc.)
                SAMESIDE (B-B/B-L/L-L): Buildings/land on same street segment side, no touching boundaries between them
                - node_1, node_2, type
                CROSSING (B-B/B-L/L-L): Buildings/land on same street segment but on opposite street segment sides
                - node_1, node_2, type, offset: none/slight/major
                NEAR (T/F-B/L): Trees/facilities to same-side buildings/land
                - node_1, node_2, type, distance_meter: 1-20 (can catch approximate spatial dimension)
                - node_1, node_2, type, relative_position_description: specific_description (be meaningful and specific for recognizing the spatial relationship)

                B-2. NODE TYPES & ATTRIBUTES
                INTERSECTION (I) - Street junction points
                id: I1, I2, I3...
                type: intersection
                linked_segment_number: 2/3/4/5/>5/unknown

                STREET SEGMENT (S) - Each physical street arm extending from an intersection, create separate segment nodes for each street direction
                id: S1, S2, S3...
                type: street segment
                street_name: accurate value or unknown if not visible
                lane_number: 1/2/3/4/5/>5/unknown
                sidewalk_description: accurate description or unknown

                BUILDING (B) - All structures facing streets
                id: B1, B2, B3...
                type: building
                building_type: accurate value or unknown
                has_ground_retail: yes/no/unknown
                story_number: 1/2/3/4/5/>5/unknown
                width_description: accurate description or unknown
                color: accurate value or unknown
                material: accurate value or unknown
                distinctive_feature: accurate description or unknown

                LAND (L) - Open spaces/lots facing streets, street or road not included
                id: L1, L2, L3...
                type: land
                land_type: accurate value or unknown
                surface: accurate value or unknown
                width_description: accurate description or unknown
                distinctive_feature: accurate description or unknown

                TREE (T) - Street trees along streets/sidewalks
                id: T1, T2, T3...
                type: tree
                size_description: accurate description or unknown
                distinctive_feature: accurate description or unknown

                FACILITY (F) - Street facilities along streets/sidewalks
                id: F1, F2, F3...
                type: facility
                facility_type: accurate value or unknown
                distinctive_feature: accurate description or unknown
                
                B. EVALUATION REQUIREMENTS:
                Based on the above CRITERIA and REFERENCE SHEET, give a score for the surveyor's json content, and provide a specific description of the evaluation.
                Use the grounding truth json if provided to score the surveyor's json content.  
                If the json score is below 100, provide a specific suggestion for SURVEY PROMPT refinement.

                EVALUATION OUTPUT FORMAT:
                {
                    "score": 0-100,
                    "evaluation": "specific description for the json content quality"
                    "refine_suggestion": "specific suggestion for prompt refinement"
                }

                Finally output the evaluation result, only valid JSON in plain text - no explanatory text or code block.
                """,

                "compare_svi_selfcheck": """
                The above json content is generated by a urban surveyor for depicting the spatial relationships for a the given street view images.
                The urban surveyor follows the SURVEY PROMPT to analyze the image and generates json content for creating structured graph representations of the street views.
                Read the SURVEY PROMPT and evaluate the surveyor's json content regarding its fullfillment of the survey requirements.
                Use the grounding truth json if provided to score the surveyor's json content.  
                If the json score is below 100, provide a specific suggestion for SURVEY PROMPT refinement.

                EVALUATION OUTPUT FORMAT:
                {
                    "score": 0-100,
                    "evaluation": "specific description for the json content quality"
                    "refine_suggestion": "specific suggestion for prompt refinement"
                }

                Finally output the evaluation result, only valid JSON in plain text - no explanatory text or code block.                

                """,

                "check_json_format": """

                    Fix this malformed JSON to match the required structure:

                    {
                    "nodes": [...],
                    "edges": [...]
                    }

                    Common fixes needed:
                    - Remove ```json markers
                    - Fix trailing commas
                    - Add missing quotes (all values should be String)
                    - Ensure proper brackets

                OUTPUT FORMAT
                - No annotation comments (such as // or #) anywhere in the JSON.
                - No explanatory text or code block, only valid JSON in plain text.            

                SAMPLE OUTPUT
                {"nodes":[{"id":"I1","type":"intersection","linked_segment_number":"4", "feature": "detailed description"}], "edges":[{"node_1":"B1","node_2":"S1","type":"ALONG"}]}

                """
            },

            "attach_content": {
                "attach_truth_json": """
                -------------------- GROUNDING TRUTH JSON ---------------------
                The grounding_truth json content is attached below, use it as a reference to score the surveyor's json content.
                """,

                "attach_survey_prompt": """
                ------------------------ SURVEY PROMPT ------------------------
                The survey prompt is attached below, use it as a reference to score the surveyor's json content and offer suggestions for prompt refinement.
                """,
            },

            # Specialized Analysis
            "refine_prompt": {
                "survey_prompt": """
                The evaluation result of the survey prompt is attached below.
                Read the "evaluation" and "refine_suggestion" to refine the "original_prompt".
                Output the refined survey prompt that can be provided as text prompt to LLM surveyor.
                Output only plain text of the refined prompt - no explanatory text or code block.
                """,
                
                "data_extraction": """
                Extract all textual and numerical data from this image:
                1. All visible text (transcribe exactly)
                2. All numbers and measurements
                3. Labels, titles, and captions
                4. Organize information systematically
                """
            },
            
            # Comparison and Follow-up
            "followup": {
                "detailed_analysis": "Can you provide more detailed analysis of the specific aspects you mentioned?",
                "clarification": "Can you clarify or expand on the points that seem unclear?",
                "comparison": "How does this compare to typical examples in this category?",
                "validation": "Can you double-check your analysis and highlight any uncertainties?"
            }
        }
    
    def get_prompt(self, category: str, prompt_name: str, **kwargs) -> str:
        """
        Get a specific prompt by category and name with optional string replacement.
        
        Args:
            category: Prompt category (e.g., 'survey', 'verification')
            prompt_name: Specific prompt name within category
            **kwargs: Key-value pairs for string replacement (e.g., city='Tokyo')
            
        Returns:
            Prompt text with placeholders replaced
        """
        try:
            prompt = self._prompts[category][prompt_name].strip()
            if not kwargs:
                return prompt
            
            # Use safe string replacement to handle JSON braces
            result = prompt
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result
        except KeyError:
            raise ValueError(f"Prompt not found: {category}.{prompt_name}")
    
    def list_categories(self) -> list:
        """List all available prompt categories."""
        return list(self._prompts.keys())
    
    def list_prompts(self, category: str) -> list:
        """List all prompts in a specific category."""
        if category not in self._prompts:
            raise ValueError(f"Category not found: {category}")
        return list(self._prompts[category].keys())
    
    def get_all_prompts(self, category: str) -> dict:
        """Get all prompts in a category."""
        if category not in self._prompts:
            raise ValueError(f"Category not found: {category}")
        return self._prompts[category].copy()
    
    def add_prompt(self, category: str, prompt_name: str, prompt_text: str) -> None:
        """Add a new prompt to a category."""
        if category not in self._prompts:
            self._prompts[category] = {}
        self._prompts[category][prompt_name] = prompt_text.strip()
    
    def search_prompts(self, keyword: str) -> list:
        """Search for prompts containing a keyword."""
        results = []
        for category, prompts in self._prompts.items():
            for name, text in prompts.items():
                if keyword.lower() in text.lower() or keyword.lower() in name.lower():
                    results.append((category, name, text))
        return results

if __name__ == "__main__":
    prompt_manager = PromptManager()
    
    # Example usage
    prompt = prompt_manager.get_prompt("svi_att", "axis_focus", city="Tokyo")
    print("Example with city replacement:")
    print(prompt)
    
