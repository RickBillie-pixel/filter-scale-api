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
    title="Debug Filter API - Vector Drawing Compatible",
    description="Debug version to find why no lines are returned",
    version="3.2.0-debug"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models remain the same...
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

class CleanPoint(BaseModel):
    x: float
    y: float

class CleanLine(BaseModel):
    p1: CleanPoint
    p2: CleanPoint
    length: float
    orientation: str
    midpoint: CleanPoint

class CleanText(BaseModel):
    text: str
    position: CleanPoint
    bounding_box: List[float]

class RegionData(BaseModel):
    label: str
    lines: List[CleanLine]
    texts: List[CleanText]

class CleanOutput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

def calculate_orientation(p1: List[float], p2: List[float], angle: Optional[float] = None) -> str:
    """Calculate line orientation"""
    if angle is not None:
        normalized_angle = abs(angle % 180)
        if normalized_angle < 15 or normalized_angle > 165:
            return "horizontal"
        elif 75 < normalized_angle < 105:
            return "vertical"
        else:
            return "diagonal"
    else:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        if dx < 1:
            return "vertical"
        elif dy < 1:
            return "horizontal"
        else:
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 15 or angle_deg > 165:
                return "horizontal"
            elif 75 < angle_deg < 105:
                return "vertical"
            else:
                return "diagonal"

def calculate_midpoint(p1: List[float], p2: List[float]) -> CleanPoint:
    """Calculate midpoint of a line"""
    return CleanPoint(
        x=round((p1[0] + p2[0]) / 2, 1),
        y=round((p1[1] + p2[1]) / 2, 1)
    )

def is_dimension_text(text: str) -> bool:
    """
    Determine if a text string represents a dimension measurement.
    Returns True if text contains dimensional information.
    """
    if not text or not text.strip():
        return False
    
    text_clean = text.strip()
    
    # 1. Must contain at least one number
    if not re.search(r'\d', text_clean):
        return False
    
    # 2. Exclude labels and room names
    exclude_patterns = [
        r'(?i)(slaapkamer|overloop|badkamer|zolder|installatie|wm|hwa|wtw|rm|balustrade|geen\s+bovenlicht)',
        r'(?i)(keuken|woonkamer|garage|berging|toilet|gang|hal|vide|dak|kelder)',
        r'(?i)(deur|raam|venster|trap|schuifdeur|openslaande)'
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, text_clean):
            return False
    
    # 3. Check for dimension patterns
    dimension_patterns = [
        r'^\d+$',                          # Pure numbers: "2050", "1000"
        r'^\d+\+vl$',                      # Door/window format: "1000+vl"
        r'^\d+(mm|cm|m)$',                 # With units: "2050mm", "105cm"
        r'^\d+x\d+(mm|cm)?$',              # Format like "94x140cm"
        r'^\d+\s?(mm|cm|m)$',              # With space: "2050 mm"
    ]
    
    is_dimension = False
    for pattern in dimension_patterns:
        if re.match(pattern, text_clean):
            is_dimension = True
            break
    
    if not is_dimension:
        return False
    
    # 4. Extract numeric value and check minimum threshold
    extracted_value = extract_dimension_value(text_clean)
    if extracted_value is None or extracted_value < 100:  # Minimum 100mm
        return False
    
    logger.debug(f"Identified dimension text: '{text_clean}' -> {extracted_value}mm")
    return True

def extract_dimension_value(text: str) -> Optional[float]:
    """
    Extract numeric dimension value from text and convert to mm.
    Returns None if no valid dimension found.
    """
    text_clean = text.strip()
    
    # Handle "94x140cm" format - take the maximum value
    match = re.match(r'^(\d+)x(\d+)(mm|cm)?$', text_clean)
    if match:
        val1, val2, unit = match.groups()
        max_val = max(int(val1), int(val2))
        
        if unit == 'cm':
            return max_val * 10  # Convert cm to mm
        elif unit == 'mm':
            return max_val
        else:
            return max_val  # Assume mm if no unit
    
    # Handle single values with units
    match = re.match(r'^(\d+)\s?(mm|cm|m|\+vl)?$', text_clean)
    if match:
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'cm':
            return value * 10  # Convert cm to mm
        elif unit == 'm':
            return value * 1000  # Convert m to mm
        else:
            return value  # Assume mm for no unit or +vl
    
    return None

def line_intersects_region(line_p1: List[float], line_p2: List[float], region: List[float]) -> bool:
    """Check if line intersects with region - with extensive debugging"""
    x1, y1, x2, y2 = region
    
    logger.debug(f"Checking line [{line_p1[0]:.1f},{line_p1[1]:.1f}] -> [{line_p2[0]:.1f},{line_p2[1]:.1f}] against region [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
    
    # Method 1: Check if any endpoint is in region
    for x, y in [line_p1, line_p2]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            logger.debug(f"  ✓ Endpoint in region: [{x:.1f},{y:.1f}]")
            return True
    
    # Method 2: Check line bounds overlap with region
    line_x1, line_y1 = line_p1
    line_x2, line_y2 = line_p2
    
    line_min_x = min(line_x1, line_x2)
    line_max_x = max(line_x1, line_x2)
    line_min_y = min(line_y1, line_y2)
    line_max_y = max(line_y1, line_y2)
    
    # Check if bounding boxes overlap
    if line_max_x < x1 or line_min_x > x2 or line_max_y < y1 or line_min_y > y2:
        logger.debug(f"  ✗ Bounding boxes don't overlap")
        return False
    
    # Method 3: Use Shapely for precise intersection
    try:
        line = LineString([line_p1, line_p2])
        region_box = box(x1, y1, x2, y2)
        intersects = line.intersects(region_box)
        logger.debug(f"  Shapely result: {intersects}")
        return intersects
    except Exception as e:
        logger.warning(f"  Shapely failed: {e}")
        return False

def text_in_region(text: VectorText, region: List[float]) -> bool:
    """Check if text center is in region"""
    center_x = (text.bounding_box[0] + text.bounding_box[2]) / 2
    center_y = (text.bounding_box[1] + text.bounding_box[3]) / 2
    x1, y1, x2, y2 = region
    in_region = x1 <= center_x <= x2 and y1 <= center_y <= y2
    
    if in_region:
        logger.debug(f"Text '{text.text}' at [{center_x:.1f},{center_y:.1f}] is in region [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
    
    return in_region

def should_include_line(line: VectorLine, drawing_type: str, region_label: str) -> bool:
    """Determine if line should be included based on drawing type and region"""
    orientation = calculate_orientation(line.p1, line.p2, line.angle)
    
    if drawing_type == "plattegrond":
        # Special rules for plattegrond: filter on length ≥200pt and only horizontal/vertical
        if line.length < 200:
            logger.debug(f"  Plattegrond filter: length {line.length:.1f} < 200 = excluded")
            return False
        
        # Only include horizontal and vertical lines for plattegrond
        if orientation not in ["horizontal", "vertical"]:
            logger.debug(f"  Plattegrond filter: orientation '{orientation}' not horizontal/vertical = excluded")
            return False
        
        include = line.length >= 200
        logger.debug(f"  Plattegrond filter: length {line.length:.1f} >= 200 and orientation '{orientation}' = {include}")
        return include
    elif drawing_type == "gevelaanzicht":
        return line.length > 40
    elif drawing_type == "detailtekening":
        return line.length > 25
    elif drawing_type == "doorsnede":
        return (line.length > 30 and orientation == "vertical") or line.is_dashed
    elif drawing_type == "bestektekening":
        label_lower = region_label.lower()
        if "grond" in label_lower or "verdieping" in label_lower:
            return line.length > 50
        elif "gevel" in label_lower:
            return line.length > 40
        elif "doorsnede" in label_lower:
            return line.length > 30 and orientation == "vertical"
        else:
            return line.length > 25
    elif drawing_type == "installatietekening":
        return line.stroke_width <= 1 or line.is_dashed
    else:
        return line.length > 10

def should_include_text(text: VectorText, drawing_type: str, region_label: str) -> bool:
    """Determine if text should be included based on drawing type and content"""
    
    if drawing_type == "plattegrond":
        # For plattegrond, prioritize dimension texts
        if is_dimension_text(text.text):
            logger.debug(f"  Plattegrond: including dimension text '{text.text}'")
            return True
        else:
            # Still include other texts but log them as non-dimension
            logger.debug(f"  Plattegrond: including non-dimension text '{text.text}'")
            return True
    
    # For other drawing types, include all texts (existing behavior)
    return True

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
                logger.debug(f"  Removing duplicate line: [{line.p1[0]:.1f},{line.p1[1]:.1f}] -> [{line.p2[0]:.1f},{line.p2[1]:.1f}]")
                break
        
        if not is_duplicate:
            unique_lines.append(line)
    
    logger.info(f"Removed {len(lines) - len(unique_lines)} duplicate lines ({len(lines)} -> {len(unique_lines)})")
    return unique_lines

def convert_vector_drawing_api_format(raw_vector_data: Dict) -> VectorData:
    """Convert Vector Drawing API format to our internal format with extensive debugging"""
    try:
        logger.info("=== Converting Vector Drawing API format ===")
        logger.debug(f"Raw data keys: {list(raw_vector_data.keys())}")
        
        pages = raw_vector_data.get("pages", [])
        if not pages:
            raise ValueError("No pages found in vector data")
        
        logger.info(f"Found {len(pages)} pages")
        
        converted_pages = []
        
        for page_idx, page_data in enumerate(pages):
            logger.debug(f"Page {page_idx} keys: {list(page_data.keys())}")
            
            page_size = page_data.get("page_size", {"width": 595.0, "height": 842.0})
            logger.info(f"Page size: {page_size}")
            
            # Extract texts
            texts = []
            raw_texts = page_data.get("texts", [])
            logger.info(f"Found {len(raw_texts)} texts")
            
            for i, text_data in enumerate(raw_texts[:3]):  # Log first 3
                logger.debug(f"Text {i}: {text_data}")
            
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
            
            # Extract lines - THIS IS THE CRITICAL PART
            lines = []
            
            # Try multiple possible locations for lines
            possible_line_locations = [
                page_data.get("drawings", {}).get("lines", []),
                page_data.get("lines", []),
                page_data.get("paths", []),
                page_data.get("elements", [])
            ]
            
            # Also check if drawings is a list
            drawings = page_data.get("drawings", {})
            if isinstance(drawings, list):
                logger.info("Drawings is a list, not a dict!")
                possible_line_locations.append(drawings)
            
            logger.info(f"Drawings type: {type(drawings)}")
            if isinstance(drawings, dict):
                logger.info(f"Drawings keys: {list(drawings.keys())}")
            
            # Find lines in any of the possible locations
            found_lines = False
            for location_idx, line_list in enumerate(possible_line_locations):
                if line_list:
                    logger.info(f"Found {len(line_list)} lines in location {location_idx}")
                    found_lines = True
                    
                    for i, line_data in enumerate(line_list[:5]):  # Log first 5
                        logger.debug(f"Line {i}: {line_data}")
                    
                    for line_data in line_list:
                        # Handle different line formats
                        if isinstance(line_data, dict):
                            # Check if it's a line
                            if line_data.get("type") == "line" or "p1" in line_data:
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
                                    # Calculate length if not provided
                                    dx = p2_list[0] - p1_list[0]
                                    dy = p2_list[1] - p1_list[1]
                                    length = math.sqrt(dx*dx + dy*dy)
                                
                                width = float(line_data.get("width", line_data.get("stroke_width", 1.0)))
                                color = line_data.get("color", [0, 0, 0])
                                
                                line = VectorLine(
                                    p1=p1_list,
                                    p2=p2_list,
                                    stroke_width=width,
                                    length=length,
                                    color=color,
                                    is_dashed=line_data.get("is_dashed", False),
                                    angle=line_data.get("angle")
                                )
                                lines.append(line)
                    
                    break  # Stop after finding lines in one location
            
            if not found_lines:
                logger.warning("NO LINES FOUND IN ANY EXPECTED LOCATION!")
                logger.info("Full page data structure:")
                logger.info(json.dumps(page_data, indent=2, default=str)[:1000] + "...")
            
            page = VectorPage(
                page_size=page_size,
                lines=lines,
                texts=texts
            )
            converted_pages.append(page)
        
        result = VectorData(
            page_number=1,
            pages=converted_pages
        )
        
        total_lines = sum(len(page.lines) for page in converted_pages)
        total_texts = sum(len(page.texts) for page in converted_pages)
        
        logger.info(f"✅ Converted Vector Drawing API data: {total_lines} lines, {total_texts} texts")
        
        if total_lines == 0:
            logger.error("⚠️ WARNING: NO LINES FOUND AFTER CONVERSION!")
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting Vector Drawing API format: {e}", exc_info=True)
        raise ValueError(f"Failed to convert Vector Drawing API format: {str(e)}")

@app.post("/filter/", response_model=CleanOutput)
async def filter_clean(input_data: FilterInput, debug: bool = Query(False)):
    """Filter data and return clean, focused output per region"""
    
    try:
        if not input_data.vector_data.pages:
            raise HTTPException(status_code=400, detail="No pages in vector_data")
        
        vector_page = input_data.vector_data.pages[0]
        drawing_type = input_data.vision_output.drawing_type
        regions = input_data.vision_output.regions
        
        logger.info(f"=== FILTERING START ===")
        logger.info(f"Processing {drawing_type} with {len(regions)} regions")
        logger.info(f"Input: {len(vector_page.lines)} lines, {len(vector_page.texts)} texts")
        
        if len(vector_page.lines) == 0:
            logger.error("NO LINES IN INPUT DATA!")
        
        # Apply plattegrond-specific preprocessing
        processed_lines = vector_page.lines
        if drawing_type == "plattegrond":
            logger.info(f"Applying plattegrond-specific preprocessing...")
            
            # Step 1: Filter by length first (≥100pt)
            length_filtered_lines = [line for line in processed_lines if line.length >= 100]
            logger.info(f"After length filter (≥100pt): {len(length_filtered_lines)} lines (from {len(processed_lines)})")
            
            # Step 2: Filter by orientation (only horizontal/vertical)
            orientation_filtered_lines = []
            for line in length_filtered_lines:
                orientation = calculate_orientation(line.p1, line.p2, line.angle)
                if orientation in ["horizontal", "vertical"]:
                    orientation_filtered_lines.append(line)
            logger.info(f"After orientation filter (horizontal/vertical): {len(orientation_filtered_lines)} lines (from {len(length_filtered_lines)})")
            
            # Step 3: Remove duplicate lines (now much fewer lines to check)
            processed_lines = remove_duplicate_lines(orientation_filtered_lines)
            
            logger.info(f"After preprocessing: {len(processed_lines)} lines")
        
        # Log first few lines and regions for debugging
        if processed_lines:
            logger.info("First 3 lines:")
            for i, line in enumerate(processed_lines[:3]):
                logger.info(f"  Line {i}: [{line.p1[0]:.1f},{line.p1[1]:.1f}] -> [{line.p2[0]:.1f},{line.p2[1]:.1f}], length={line.length:.1f}")
        
        logger.info("Regions:")
        for region in regions:
            logger.info(f"  {region.label}: {region.coordinate_block}")
        
        region_outputs = []
        total_lines_checked = 0
        total_lines_in_regions = 0
        total_lines_included = 0
        total_dimension_texts = 0
        
        for region in regions:
            region_lines = []
            region_texts = []
            lines_in_this_region = 0
            dimension_texts_in_region = 0
            
            logger.info(f"\nProcessing region: {region.label}")
            logger.info(f"  Region bounds: {region.coordinate_block}")
            
            # Process lines for this region
            for i, line in enumerate(processed_lines):
                total_lines_checked += 1
                
                # Check if line is in region
                is_in_region = line_intersects_region(line.p1, line.p2, region.coordinate_block)
                
                if is_in_region:
                    lines_in_this_region += 1
                    total_lines_in_regions += 1
                    
                    # Check if line should be included based on rules
                    should_include = should_include_line(line, drawing_type, region.label)
                    
                    if should_include:
                        total_lines_included += 1
                        clean_line = CleanLine(
                            p1=CleanPoint(x=round(line.p1[0], 1), y=round(line.p1[1], 1)),
                            p2=CleanPoint(x=round(line.p2[0], 1), y=round(line.p2[1], 1)),
                            length=round(line.length, 1),
                            orientation=calculate_orientation(line.p1, line.p2, line.angle),
                            midpoint=calculate_midpoint(line.p1, line.p2)
                        )
                        region_lines.append(clean_line)
            
            # Process texts for this region
            texts_in_region = 0
            for text in vector_page.texts:
                if text_in_region(text, region.coordinate_block):
                    texts_in_region += 1
                    
                    # Check if text should be included
                    if should_include_text(text, drawing_type, region.label):
                        # Check if it's a dimension text for statistics
                        if drawing_type == "plattegrond" and is_dimension_text(text.text):
                            dimension_texts_in_region += 1
                            total_dimension_texts += 1
                        
                        clean_text = CleanText(
                            text=text.text,
                            position=CleanPoint(x=round(text.position[0], 1), y=round(text.position[1], 1)),
                            bounding_box=[round(x, 1) for x in text.bounding_box]
                        )
                        region_texts.append(clean_text)
            
            # Create region output
            region_data = RegionData(
                label=region.label,
                lines=region_lines,
                texts=region_texts
            )
            region_outputs.append(region_data)
            
            logger.info(f"  {region.label} results:")
            logger.info(f"    Lines in region: {lines_in_this_region}")
            logger.info(f"    Lines included: {len(region_lines)}")
            logger.info(f"    Texts in region: {texts_in_region}")
            if drawing_type == "plattegrond":
                logger.info(f"    Dimension texts: {dimension_texts_in_region}")
        
        # Create clean output
        output = CleanOutput(
            drawing_type=drawing_type,
            regions=region_outputs
        )
        
        logger.info(f"\n=== FILTERING COMPLETE ===")
        logger.info(f"Total lines checked: {total_lines_checked}")
        logger.info(f"Total lines in any region: {total_lines_in_regions}")
        logger.info(f"Total lines included: {total_lines_included}")
        logger.info(f"Total texts: {sum(len(r.texts) for r in region_outputs)}")
        if drawing_type == "plattegrond":
            logger.info(f"Total dimension texts identified: {total_dimension_texts}")
        
        if total_lines_included == 0:
            logger.error("⚠️ NO LINES INCLUDED IN OUTPUT!")
            logger.error("Possible issues:")
            logger.error("1. No lines in input data")
            logger.error("2. Region coordinates don't match line coordinates")
            logger.error("3. All lines filtered out by length requirements")
            if drawing_type == "plattegrond":
                logger.error("4. All lines filtered out by plattegrond rules (length < 100 or not horizontal/vertical)")
        
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
        "service": "Debug Filter API",
        "version": "3.2.0-debug"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Debug Filter API",
        "version": "3.2.0-debug",
        "description": "Debug version to find why no lines are returned",
        "debug_features": [
            "Extensive logging of line processing",
            "Multiple line location checks",
            "Detailed intersection debugging",
            "Line count tracking at each stage"
        ],
        "new_features": [
            "Plattegrond-specific filtering (length ≥100pt, horizontal/vertical only)",
            "Dimension text detection with regex patterns",
            "Duplicate line removal for plattegrond",
            "Enhanced text filtering for dimensional data"
        ],
        "endpoints": {
            "/filter/": "Main filtering endpoint",
            "/filter-from-vector-api/": "Direct endpoint for Vector Drawing API output",
            "/health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
