import os
import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import LineString, box
import math

# Configure logging with more detail
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Filter API v7.0.0 - Vision Compatible Bestektekening",
    description="Updated for Vision's new bestektekening region format with drawing type classification",
    version="7.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input models
class VectorLine(BaseModel):
    p1: List[float]
    p2: List[float] 
    stroke_width: float = Field(default=1.0)
    length: float
    color: List[int] = Field(default=[0, 0, 0])
    is_dashed: bool = Field(default=False)
    angle: Optional[float] = None

class VectorText(BaseModel):
    text: str
    position: List[float]
    font_size: Optional[float] = None
    bounding_box: List[float]

class VectorPage(BaseModel):
    page_size: Dict[str, float]
    lines: List[VectorLine]
    texts: List[VectorText]

class VectorData(BaseModel):
    page_number: int
    pages: List[VectorPage]

class VisionRegion(BaseModel):
    label: str
    coordinate_block: List[float]

class VisionOutput(BaseModel):
    drawing_type: str
    regions: List[VisionRegion]
    image_metadata: Optional[Dict] = None

class FilterInput(BaseModel):
    vector_data: VectorData
    vision_output: VisionOutput

# Output models - optimized for Scale API
class CleanPoint(BaseModel):
    x: float
    y: float

class FilteredLine(BaseModel):
    length: float
    orientation: str
    midpoint: CleanPoint

class FilteredText(BaseModel):
    text: str
    midpoint: Dict[str, float]
    
class RegionData(BaseModel):
    label: str
    lines: List[FilteredLine]
    texts: List[FilteredText]
    parsed_drawing_type: Optional[str] = None  # NEW: Extracted drawing type for bestektekening

class CleanOutput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

def parse_bestektekening_region_type(region_label: str) -> str:
    """
    Extract drawing type from bestektekening region label
    Handles new Vision format: "Begane grond (plattegrond)" → "plattegrond"
    """
    
    # Check for explicit type in parentheses first
    if "(" in region_label and ")" in region_label:
        try:
            start = region_label.find("(") + 1
            end = region_label.find(")")
            extracted_type = region_label[start:end].strip()
            
            # Validate extracted type
            valid_types = [
                "plattegrond", "doorsnede", "gevelaanzicht", 
                "detailtekening_kozijn", "detailtekening_plattegrond",
                "detailtekening"
            ]
            
            if extracted_type in valid_types:
                logger.debug(f"Extracted drawing type '{extracted_type}' from label '{region_label}'")
                return extracted_type
                
        except Exception as e:
            logger.warning(f"Failed to parse parentheses in label '{region_label}': {e}")
    
    # Fallback to keyword matching (legacy compatibility)
    label_lower = region_label.lower()
    
    if "plattegrond" in label_lower or "grond" in label_lower or "verdieping" in label_lower:
        logger.debug(f"Fallback: detected plattegrond from '{region_label}'")
        return "plattegrond"
    elif "gevel" in label_lower or "aanzicht" in label_lower:
        logger.debug(f"Fallback: detected gevelaanzicht from '{region_label}'")
        return "gevelaanzicht"
    elif "doorsnede" in label_lower:
        logger.debug(f"Fallback: detected doorsnede from '{region_label}'")
        return "doorsnede"
    elif "detail" in label_lower:
        if "kozijn" in label_lower or "raam" in label_lower or "deur" in label_lower:
            logger.debug(f"Fallback: detected detailtekening_kozijn from '{region_label}'")
            return "detailtekening_kozijn"
        else:
            logger.debug(f"Fallback: detected detailtekening (generic) from '{region_label}'")
            return "detailtekening"
    else:
        logger.debug(f"No drawing type detected from '{region_label}', using unknown")
        return "unknown"

def calculate_orientation(p1: List[float], p2: List[float], angle: Optional[float] = None) -> str:
    """Calculate line orientation"""
    if angle is not None:
        normalized_angle = abs(angle % 180)
        if normalized_angle < 15 or normalized_angle > 165:
            return "vertical"
        elif 75 < normalized_angle < 105:
            return "horizontal"
        else:
            return "diagonal"
    else:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        if dx < 1:
            return "horizontal"
        elif dy < 1:
            return "vertical"
        else:
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 15 or angle_deg > 165:
                return "vertical"
            elif 75 < angle_deg < 105:
                return "horizontal"
            else:
                return "diagonal"

def calculate_midpoint(p1: List[float], p2: List[float]) -> CleanPoint:
    """Calculate midpoint of a line"""
    return CleanPoint(
        x=(p1[0] + p2[0]) / 2,
        y=(p1[1] + p2[1]) / 2
    )

def calculate_text_midpoint(bbox: List[float]) -> Dict[str, float]:
    """Calculate midpoint of text bounding box"""
    return {
        "x": (bbox[0] + bbox[2]) / 2,
        "y": (bbox[1] + bbox[3]) / 2
    }

def extract_dimension_value(text: str) -> Optional[float]:
    """
    Extract numeric dimension value from text and convert to mm.
    Returns None if no valid dimension found.
    """
    text_clean = text.strip()
    
    # Handle decimal numbers with comma or dot
    match = re.match(r'^(\d+(?:[,.]\d+)?)\s*(mm|cm|m)?$', text_clean)
    if match:
        value_str, unit = match.groups()
        # Replace comma with dot for parsing
        value_str = value_str.replace(',', '.')
        value = float(value_str)
        
        if unit == 'cm':
            return value * 10  # Convert cm to mm
        elif unit == 'm':
            return value * 1000  # Convert m to mm
        else:
            return value  # Assume mm if no unit
    
    return None

def is_valid_dimension(text: str, drawing_type: str = "general") -> bool:
    """
    Validate if text represents a pure dimension (numbers with optional units).
    Excludes texts with letters, symbols, or non-dimensional content.
    """
    if not text or not text.strip():
        return False
    
    text_clean = text.strip()
    
    # Pattern for valid dimensions: pure numbers with optional units
    # Matches: "3000", "3000mm", "3,5 m", "250 cm", "3.5m"
    pattern = r'^\d+([,.]\d+)?\s*(mm|cm|m)?$'
    
    is_valid = bool(re.match(pattern, text_clean))
    
    if not is_valid:
        logger.debug(f"Invalid dimension text: '{text_clean}' (contains letters/symbols or invalid format)")
        return False
    
    # Extract numeric value to check minimum threshold
    extracted_value = extract_dimension_value(text_clean)
    if extracted_value is None:
        logger.debug(f"Could not extract numeric value from: '{text_clean}'")
        return False
    
    # Apply drawing-type specific minimum thresholds
    if drawing_type == "plattegrond":
        # For plattegrond: minimum 500mm
        min_value = 500
        if extracted_value < min_value:
            logger.debug(f"Plattegrond filter: '{text_clean}' = {extracted_value}mm < {min_value}mm = excluded")
            return False
        else:
            logger.debug(f"Valid plattegrond dimension: '{text_clean}' = {extracted_value}mm >= {min_value}mm")
            return True
    else:
        # For other drawing types: minimum 100mm
        min_value = 100
        if extracted_value < min_value:
            logger.debug(f"General filter: '{text_clean}' = {extracted_value}mm < {min_value}mm = excluded")
            return False
        else:
            logger.debug(f"Valid dimension: '{text_clean}' = {extracted_value}mm >= {min_value}mm")
            return True

def line_intersects_region(line_p1: List[float], line_p2: List[float], region: List[float]) -> bool:
    """Check if line intersects with expanded region (25pt buffer)"""
    x1, y1, x2, y2 = region
    # 25pt buffer for lines
    expanded_region = [x1 - 25, y1 - 25, x2 + 25, y2 + 25]
    
    # Method 1: Check if any endpoint is in expanded region
    for x, y in [line_p1, line_p2]:
        if expanded_region[0] <= x <= expanded_region[2] and expanded_region[1] <= y <= expanded_region[3]:
            return True
    
    # Method 2: Check line bounds overlap with expanded region
    line_x1, line_y1 = line_p1
    line_x2, line_y2 = line_p2
    
    line_min_x = min(line_x1, line_x2)
    line_max_x = max(line_x1, line_x2)
    line_min_y = min(line_y1, line_y2)
    line_max_y = max(line_y1, line_y2)
    
    # Check if bounding boxes overlap
    if line_max_x < expanded_region[0] or line_min_x > expanded_region[2] or line_max_y < expanded_region[1] or line_min_y > expanded_region[3]:
        return False
    
    # Method 3: Use Shapely for precise intersection
    try:
        line = LineString([line_p1, line_p2])
        region_box = box(expanded_region[0], expanded_region[1], expanded_region[2], expanded_region[3])
        return line.intersects(region_box)
    except Exception as e:
        logger.warning(f"Shapely intersection failed: {e}")
        return False

def text_overlaps_region(text: VectorText, region: List[float]) -> bool:
    """Check if text bounding box overlaps with expanded region (25pt buffer)"""
    # 25pt buffer
    x1, y1, x2, y2 = region
    expanded_region = [x1 - 25, y1 - 25, x2 + 25, y2 + 25]
    
    # Text bounding box
    text_x1, text_y1, text_x2, text_y2 = text.bounding_box
    
    # Check if bounding boxes overlap
    return not (text_x2 < expanded_region[0] or  # text is left of region
                text_x1 > expanded_region[2] or  # text is right of region
                text_y2 < expanded_region[1] or  # text is below region
                text_y1 > expanded_region[3])    # text is above region

def should_include_line(line: VectorLine, drawing_type: str, region_label: str) -> bool:
    """
    UPDATED v7.0.0: Enhanced filtering rules with bestektekening region parsing
    """
    
    # Handle bestektekening with new Vision format
    if drawing_type == "bestektekening":
        region_drawing_type = parse_bestektekening_region_type(region_label)
        logger.debug(f"Bestektekening region '{region_label}' parsed as '{region_drawing_type}'")
        
        # Apply rules for the parsed drawing type
        if region_drawing_type == "plattegrond":
            return (line.stroke_width <= 1.5 and line.length >= 50) or line.is_dashed
        elif region_drawing_type == "gevelaanzicht":
            return (line.stroke_width <= 1.5 and line.length >= 40) or line.is_dashed
        elif region_drawing_type == "doorsnede":
            return (line.stroke_width <= 1.5 and line.length >= 40) or line.is_dashed
        elif region_drawing_type in ["detailtekening_kozijn", "detailtekening_plattegrond", "detailtekening"]:
            return (line.stroke_width <= 1.0 and line.length >= 20) or line.is_dashed
        else:
            # Unknown region type - use conservative default
            logger.warning(f"Unknown region type '{region_drawing_type}' for bestektekening, using default rules")
            return (line.stroke_width <= 1.5 and line.length >= 30) or line.is_dashed
    
    # Standard drawing type rules (unchanged)
    elif drawing_type == "plattegrond":
        return (line.stroke_width <= 1.5 and line.length >= 50) or line.is_dashed
    
    elif drawing_type == "gevelaanzicht":
        return (line.stroke_width <= 1.5 and line.length >= 40) or line.is_dashed
    
    elif drawing_type == "doorsnede":
        return (line.stroke_width <= 1.5 and line.length >= 40) or line.is_dashed
    
    elif drawing_type in ["detailtekening", "detailtekening_kozijn", "detailtekening_plattegrond"]:
        return (line.stroke_width <= 1.0 and line.length >= 20) or line.is_dashed
    
    elif drawing_type == "installatietekening":
        # Skip installatietekening (no processing)
        return False
    
    else:
        # Default: Conservative rules
        return (line.stroke_width <= 1.5 and line.length >= 30) or line.is_dashed

def should_include_text(text: VectorText, drawing_type: str, region_label: str) -> bool:
    """
    UPDATED v7.0.0: Enhanced text filtering with bestektekening region parsing
    """
    
    # Handle bestektekening with region-specific rules
    if drawing_type == "bestektekening":
        region_drawing_type = parse_bestektekening_region_type(region_label)
        effective_drawing_type = region_drawing_type
    else:
        effective_drawing_type = drawing_type
    
    # Skip installatietekening
    if effective_drawing_type == "installatietekening":
        return False
    
    # Apply dimension validation
    is_valid = is_valid_dimension(text.text, effective_drawing_type)
    
    if is_valid:
        logger.debug(f"Including valid dimension text: '{text.text}' for {effective_drawing_type}")
    else:
        logger.debug(f"Excluding invalid dimension text: '{text.text}' for {effective_drawing_type}")
    
    return is_valid

def remove_duplicate_lines(lines: List[VectorLine]) -> List[VectorLine]:
    """Remove duplicate lines based on p1, p2 coordinates with small tolerance"""
    unique_lines = []
    tolerance = 1.0  # 1 point tolerance for coordinate comparison
    
    for line in lines:
        is_duplicate = False
        
        for existing_line in unique_lines:
            # Check if coordinates are within tolerance
            p1_match = (abs(line.p1[0] - existing_line.p1[0]) < tolerance and 
                       abs(line.p1[1] - existing_line.p1[1]) < tolerance)
            p2_match = (abs(line.p2[0] - existing_line.p2[0]) < tolerance and 
                       abs(line.p2[1] - existing_line.p2[1]) < tolerance)
            
            # Also check reverse direction
            p1_reverse = (abs(line.p1[0] - existing_line.p2[0]) < tolerance and 
                         abs(line.p1[1] - existing_line.p2[1]) < tolerance)
            p2_reverse = (abs(line.p2[0] - existing_line.p1[0]) < tolerance and 
                         abs(line.p2[1] - existing_line.p1[1]) < tolerance)
            
            if (p1_match and p2_match) or (p1_reverse and p2_reverse):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_lines.append(line)
    
    logger.info(f"Removed {len(lines) - len(unique_lines)} duplicate lines ({len(lines)} -> {len(unique_lines)})")
    return unique_lines

def convert_vector_drawing_api_format(raw_vector_data: Dict) -> VectorData:
    """Convert Vector Drawing API format to our internal format"""
    try:
        logger.info("=== Converting Vector Drawing API format ===")
        
        pages = raw_vector_data.get("pages", [])
        if not pages:
            raise ValueError("No pages found in vector data")
        
        logger.info(f"Found {len(pages)} pages")
        converted_pages = []
        
        for page_idx, page_data in enumerate(pages):
            page_size = page_data.get("page_size", {"width": 595.0, "height": 842.0})
            
            # Extract texts
            texts = []
            raw_texts = page_data.get("texts", [])
            logger.info(f"Found {len(raw_texts)} texts")
            
            for text_data in raw_texts:
                position = text_data.get("position", {"x": 0, "y": 0})
                bbox = text_data.get("bbox", {})
                
                if isinstance(position, dict):
                    pos_list = [float(position.get("x", 0)), float(position.get("y", 0))]
                else:
                    pos_list = [float(position[0]), float(position[1])]
                
                if isinstance(bbox, dict):
                    bbox_list = [
                        float(bbox.get("x0", 0)), 
                        float(bbox.get("y0", 0)), 
                        float(bbox.get("x1", 100)), 
                        float(bbox.get("y1", 20))
                    ]
                else:
                    bbox_list = [float(x) for x in bbox[:4]] if len(bbox) >= 4 else [0, 0, 100, 20]
                
                text = VectorText(
                    text=text_data.get("text", ""),
                    position=pos_list,
                    font_size=text_data.get("font_size", 12.0),
                    bounding_box=bbox_list
                )
                texts.append(text)
            
            # Extract lines
            lines = []
            possible_line_locations = [
                page_data.get("drawings", {}).get("lines", []),
                page_data.get("lines", []),
                page_data.get("paths", []),
                page_data.get("elements", [])
            ]
            
            drawings = page_data.get("drawings", {})
            if isinstance(drawings, list):
                possible_line_locations.append(drawings)
            
            # Find lines in any of the possible locations
            found_lines = False
            for location_idx, line_list in enumerate(possible_line_locations):
                if line_list:
                    logger.info(f"Found {len(line_list)} lines in location {location_idx}")
                    found_lines = True
                    
                    for line_data in line_list:
                        if isinstance(line_data, dict) and (line_data.get("type") == "line" or "p1" in line_data):
                            p1 = line_data.get("p1", {"x": 0, "y": 0})
                            p2 = line_data.get("p2", {"x": 0, "y": 0})
                            
                            if isinstance(p1, dict):
                                p1_list = [float(p1.get("x", 0)), float(p1.get("y", 0))]
                            else:
                                p1_list = [float(p1[0]), float(p1[1])]
                            
                            if isinstance(p2, dict):
                                p2_list = [float(p2.get("x", 0)), float(p2.get("y", 0))]
                            else:
                                p2_list = [float(p2[0]), float(p2[1])]
                            
                            length = float(line_data.get("length", 0))
                            if length == 0:
                                dx = p2_list[0] - p1_list[0]
                                dy = p2_list[1] - p1_list[1]
                                length = math.sqrt(dx*dx + dy*dy)
                            
                            line = VectorLine(
                                p1=p1_list,
                                p2=p2_list,
                                stroke_width=float(line_data.get("width", line_data.get("stroke_width", 1.0))),
                                length=length,
                                color=line_data.get("color", [0, 0, 0]),
                                is_dashed=line_data.get("is_dashed", False),
                                angle=line_data.get("angle")
                            )
                            lines.append(line)
                    break
            
            if not found_lines:
                logger.warning("NO LINES FOUND IN ANY EXPECTED LOCATION!")
            
            page = VectorPage(
                page_size=page_size,
                lines=lines,
                texts=texts
            )
            converted_pages.append(page)
        
        result = VectorData(page_number=1, pages=converted_pages)
        
        total_lines = sum(len(page.lines) for page in converted_pages)
        total_texts = sum(len(page.texts) for page in converted_pages)
        logger.info(f"✅ Converted Vector Drawing API data: {total_lines} lines, {total_texts} texts")
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting Vector Drawing API format: {e}", exc_info=True)
        raise ValueError(f"Failed to convert Vector Drawing API format: {str(e)}")

@app.post("/filter/", response_model=CleanOutput)
async def filter_clean(input_data: FilterInput, debug: bool = Query(False)):
    """Filter data and return clean, Scale API compatible output per region"""
    
    try:
        if not input_data.vector_data.pages:
            raise HTTPException(status_code=400, detail="No pages in vector_data")
        
        vector_page = input_data.vector_data.pages[0]
        drawing_type = input_data.vision_output.drawing_type
        regions = input_data.vision_output.regions
        
        logger.info(f"=== FILTERING START (v7.0.0 - Optimized) ===")
        logger.info(f"Processing {drawing_type} with {len(regions)} regions")
        logger.info(f"Input: {len(vector_page.lines)} lines, {len(vector_page.texts)} texts")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"25pt buffer applied to all regions")
        
        # Skip installatietekening entirely
        if drawing_type == "installatietekening":
            logger.info("Skipping installatietekening - no processing")
            return CleanOutput(drawing_type=drawing_type, regions=[])
        
        # OPTIMIZED: Apply filtering FIRST, then duplicate removal for plattegrond
        processed_lines = vector_page.lines
        if drawing_type == "plattegrond":
            logger.info(f"Applying plattegrond-specific preprocessing...")
            
            # STEP 1: Pre-filter lines based on plattegrond rules (MUCH faster)
            pre_filtered_lines = []
            plattegrond_rule_applied = 0
            
            for line in vector_page.lines:
                # Apply plattegrond line rules first
                if (line.stroke_width <= 1.5 and line.length >= 50) or line.is_dashed:
                    pre_filtered_lines.append(line)
                    plattegrond_rule_applied += 1
            
            logger.info(f"After plattegrond rules filter: {len(pre_filtered_lines)} lines (was {len(vector_page.lines)})")
            logger.info(f"Filtered out {len(vector_page.lines) - len(pre_filtered_lines)} lines that didn't meet criteria")
            
            # STEP 2: Remove duplicates from the much smaller filtered set
            if len(pre_filtered_lines) > 0:
                processed_lines = remove_duplicate_lines(pre_filtered_lines)
                logger.info(f"After duplicate removal: {len(processed_lines)} lines")
            else:
                processed_lines = pre_filtered_lines
                logger.info("No lines left after filtering - skipping duplicate removal")
        else:
            # For other drawing types: no preprocessing needed
            logger.info(f"No preprocessing needed for {drawing_type}")
            processed_lines = vector_page.lines
        
        region_outputs = []
        total_lines_included = 0
        total_valid_dimension_texts = 0
        
        for region in regions:
            region_lines = []
            region_texts = []
            
            logger.info(f"\nProcessing region: {region.label}")
            
            # Parse drawing type for bestektekening regions
            parsed_drawing_type = None
            if drawing_type == "bestektekening":
                parsed_drawing_type = parse_bestektekening_region_type(region.label)
                logger.info(f"  Parsed drawing type: {parsed_drawing_type}")
            
            # Process lines for this region
            lines_in_region = 0
            lines_passed_filter = 0
            
            for line in processed_lines:
                if line_intersects_region(line.p1, line.p2, region.coordinate_block):
                    lines_in_region += 1
                    
                    # For plattegrond, lines are already pre-filtered, so just check region intersection
                    if drawing_type == "plattegrond":
                        # Lines are already filtered by plattegrond rules, just add them
                        total_lines_included += 1
                        lines_passed_filter += 1
                        filtered_line = FilteredLine(
                            length=line.length,
                            orientation=calculate_orientation(line.p1, line.p2, line.angle),
                            midpoint=calculate_midpoint(line.p1, line.p2)
                        )
                        region_lines.append(filtered_line)
                        if debug:
                            logger.debug(f"  ✅ Line included: {line.length:.1f}pt, stroke: {line.stroke_width}pt, orientation: {filtered_line.orientation}")
                    else:
                        # For other drawing types, apply filtering rules
                        if should_include_line(line, drawing_type, region.label):
                            total_lines_included += 1
                            lines_passed_filter += 1
                            filtered_line = FilteredLine(
                                length=line.length,
                                orientation=calculate_orientation(line.p1, line.p2, line.angle),
                                midpoint=calculate_midpoint(line.p1, line.p2)
                            )
                            region_lines.append(filtered_line)
                            if debug:
                                logger.debug(f"  ✅ Line included: {line.length:.1f}pt, stroke: {line.stroke_width}pt, orientation: {filtered_line.orientation}")
                        elif debug:
                            logger.debug(f"  ❌ Line excluded: {line.length:.1f}pt, stroke: {line.stroke_width}pt")
            
            logger.info(f"  Lines in region: {lines_in_region}, passed filter: {lines_passed_filter}")
            
            # Process texts for this region - 25pt buffer overlap detection
            texts_in_region = 0
            texts_passed_filter = 0
            
            for text in vector_page.texts:
                if text_overlaps_region(text, region.coordinate_block):
                    texts_in_region += 1
                    
                    if should_include_text(text, drawing_type, region.label):
                        total_valid_dimension_texts += 1
                        texts_passed_filter += 1
                        
                        # Create filtered text with midpoint
                        filtered_text = FilteredText(
                            text=text.text,
                            midpoint=calculate_text_midpoint(text.bounding_box)
                        )
                        
                        region_texts.append(filtered_text)
                        if debug:
                            logger.debug(f"  ✅ Text included: '{text.text}'")
                    elif debug:
                        logger.debug(f"  ❌ Text excluded: '{text.text}'")
            
            logger.info(f"  Texts in region: {texts_in_region}, passed filter: {texts_passed_filter}")
            
            # Create region output with parsed drawing type
            region_data = RegionData(
                label=region.label,
                lines=region_lines,
                texts=region_texts,
                parsed_drawing_type=parsed_drawing_type
            )
            region_outputs.append(region_data)
            
            logger.info(f"  {region.label} final results:")
            logger.info(f"    Lines included: {len(region_lines)}")
            logger.info(f"    Valid dimension texts: {len(region_texts)}")
            if parsed_drawing_type:
                logger.info(f"    Parsed drawing type: {parsed_drawing_type}")
        
        # Create clean output
        output = CleanOutput(
            drawing_type=drawing_type,
            regions=region_outputs
        )
        
        logger.info(f"\n=== FILTERING COMPLETE ===")
        logger.info(f"Total lines processed: {len(processed_lines)}")
        logger.info(f"Total lines included: {total_lines_included}")
        logger.info(f"Total valid dimension texts: {total_valid_dimension_texts}")
        logger.info(f"Regions processed: {len(region_outputs)}")
        
        return output
    
    except Exception as e:
        logger.error(f"Error in filter_clean: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/filter-from-vector-api/", response_model=CleanOutput)
async def filter_from_vector_api(
    vector_data: Dict[str, Any], 
    vision_output: Dict[str, Any],
    debug: bool = Query(False)
):
    """Direct endpoint that accepts raw Vector Drawing API output"""
    
    try:
        logger.info("=== Processing raw Vector Drawing API output ===")
        
        # Convert Vector Drawing API format to our internal format
        converted_vector_data = convert_vector_drawing_api_format(vector_data)
        
        # Create vision output object
        vision_obj = VisionOutput(**vision_output)
        
        # Create filter input
        filter_input = FilterInput(
            vector_data=converted_vector_data,
            vision_output=vision_obj
        )
        
        # Process using the main filter function
        return await filter_clean(filter_input, debug)
        
    except Exception as e:
        logger.error(f"Error processing Vector API output: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing Vector API output: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Filter API v7.0.0 - Vision Compatible",
        "version": "7.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Filter API v7.0.0 - Vision Compatible Bestektekening",
        "version": "7.0.0",
        "description": "Updated for Vision's new bestektekening region format with drawing type classification",
        "new_features_v7": [
            "✅ Parses drawing types from Vision region labels",
            "✅ Handles new bestektekening format: 'Begane grond (plattegrond)'",
            "✅ Fallback to keyword matching for legacy compatibility",
            "✅ Adds parsed_drawing_type field to RegionData for Scale API",
            "✅ Enhanced logging for bestektekening region processing",
            "✅ Installatietekening completely skipped"
        ],
        "parsing_examples": {
            "new_format": {
                "input": "Begane grond (plattegrond)",
                "extracted": "plattegrond",
                "rules_applied": "plattegrond filtering rules"
            },
            "legacy_format": {
                "input": "Eerste verdieping",
                "fallback": "plattegrond (keyword: verdieping)",
                "rules_applied": "plattegrond filtering rules"
            },
            "unknown_format": {
                "input": "Onbekende regio",
                "fallback": "unknown",
                "rules_applied": "default conservative rules"
            }
        },
        "bestektekening_region_rules": {
            "plattegrond": "stroke ≤1.5pt, length ≥50pt",
            "gevelaanzicht": "stroke ≤1.5pt, length ≥40pt",
            "doorsnede": "stroke ≤1.5pt, length ≥40pt", 
            "detailtekening_kozijn": "stroke ≤1.0pt, length ≥20pt",
            "detailtekening_plattegrond": "stroke ≤1.0pt, length ≥20pt",
            "unknown": "stroke ≤1.5pt, length ≥30pt (conservative)"
        },
        "filtering_rules": {
            "plattegrond": "stroke ≤1.5pt, length ≥50pt",
            "gevelaanzicht": "stroke ≤1.5pt, length ≥40pt", 
            "doorsnede": "stroke ≤1.5pt, length ≥40pt",
            "detailtekening": "stroke ≤1.0pt, length ≥20pt",
            "detailtekening_kozijn": "stroke ≤1.0pt, length ≥20pt",
            "detailtekening_plattegrond": "stroke ≤1.0pt, length ≥20pt",
            "bestektekening": "per_region_rules based on parsed drawing type",
            "installatietekening": "SKIPPED - no processing",
            "default": "stroke ≤1.5pt, length ≥30pt"
        },
        "text_filtering": {
            "valid_patterns": ["3000", "3000mm", "3,5 m", "250 cm"],
            "invalid_patterns": ["Keuken", "A-01", "Schaal 1:100", "Noord"],
            "minimum_values": {
                "plattegrond": "500mm",
                "other": "100mm"
            },
            "bestektekening": "Uses parsed region drawing type for validation"
        },
        "output_enhancements": {
            "parsed_drawing_type": "Added to RegionData for bestektekening regions",
            "scale_api_compatibility": "Ready for Scale API v7.0.0",
            "debug_logging": "Enhanced region processing visibility"
        },
        "compatibility": {
            "filter_api_v6": "Backward compatible with existing API calls",
            "scale_api_v7": "Forward compatible with new parsed_drawing_type field",
            "master_api": "Works with existing Master API v4.1.0"
        },
        "technical_specifications": {
            "buffer_size": "25pt for both lines and texts",
            "orientation_detection": "Horizontal, vertical, diagonal",
            "duplicate_removal": "For plattegrond only (1pt tolerance)",
            "region_intersection": "Shapely-based geometric intersection",
            "coordinate_formats": "Supports both dict and list formats"
        },
        "api_workflow": [
            "1. Receive Vector data (lines/texts) and Vision output (regions)",
            "2. Parse bestektekening region labels to extract drawing types",
            "3. Apply 25pt buffer expansion to all regions",
            "4. Filter lines based on stroke width and length per drawing type",
            "5. Filter texts to only include valid dimensions with units",
            "6. Calculate orientations and midpoints for all filtered elements",
            "7. Return clean output with parsed_drawing_type for Scale API"
        ],
        "endpoints": {
            "/filter/": "Main filtering endpoint (add ?debug=true for detailed output)",
            "/filter-from-vector-api/": "Direct endpoint for Vector Drawing API output",
            "/health": "Health check with timestamp",
            "/": "This documentation endpoint"
        },
        "logging_levels": {
            "INFO": "High-level processing steps and results",
            "DEBUG": "Detailed line/text inclusion/exclusion decisions",
            "WARNING": "Parsing failures and fallback usage",
            "ERROR": "Processing errors and exceptions"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Filter API v7.0.0 on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
