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
    title="Filter API - Scale API Compatible",
    description="Optimized for Scale API with midpoints, orientations, and dimension filtering",
    version="4.0.0-scale-compatible"
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
    orientation: str
    # Optional fields only included in debug mode
    position: Optional[Dict[str, float]] = None
    bounding_box: Optional[List[float]] = None

class RegionData(BaseModel):
    label: str
    lines: List[FilteredLine]
    texts: List[FilteredText]

class CleanOutput(BaseModel):
    drawing_type: str
    regions: List[RegionData]

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
                return "horizontal"  # ← FIX: was "vertical"
            else:
                return "diagonal"

def calculate_midpoint(p1: List[float], p2: List[float]) -> CleanPoint:
    """Calculate midpoint of a line"""
    return CleanPoint(
        x=round((p1[0] + p2[0]) / 2, 1),
        y=round((p1[1] + p2[1]) / 2, 1)
    )

def calculate_text_midpoint(bbox: List[float]) -> Dict[str, float]:
    """Calculate midpoint of text bounding box"""
    return {
        "x": round((bbox[0] + bbox[2]) / 2, 1),
        "y": round((bbox[1] + bbox[3]) / 2, 1)
    }

def calculate_text_orientation(bbox: List[float]) -> str:
    """Calculate text orientation based on bounding box dimensions"""
    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    return "vertical" if width >= height else "horizontal"  

def is_valid_dimension(text: str) -> bool:
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
    
    if is_valid:
        logger.debug(f"Valid dimension text: '{text_clean}'")
    else:
        logger.debug(f"Invalid dimension text: '{text_clean}' (contains letters/symbols or invalid format)")
    
    return is_valid

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

def line_intersects_region(line_p1: List[float], line_p2: List[float], region: List[float]) -> bool:
    """Check if line intersects with region"""
    x1, y1, x2, y2 = region
    
    # Method 1: Check if any endpoint is in region
    for x, y in [line_p1, line_p2]:
        if x1 <= x <= x2 and y1 <= y <= y2:
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
        return False
    
    # Method 3: Use Shapely for precise intersection
    try:
        line = LineString([line_p1, line_p2])
        region_box = box(x1, y1, x2, y2)
        return line.intersects(region_box)
    except Exception as e:
        logger.warning(f"Shapely intersection failed: {e}")
        return False

def text_in_region(text: VectorText, region: List[float]) -> bool:
    """Check if text center is in region"""
    center_x = (text.bounding_box[0] + text.bounding_box[2]) / 2
    center_y = (text.bounding_box[1] + text.bounding_box[3]) / 2
    x1, y1, x2, y2 = region
    return x1 <= center_x <= x2 and y1 <= center_y <= y2

def should_include_line(line: VectorLine, drawing_type: str, region_label: str) -> bool:
    """Determine if line should be included based on drawing type and region"""
    orientation = calculate_orientation(line.p1, line.p2, line.angle)
    
    if drawing_type == "plattegrond":
        # Plattegrond rules: length ≥100pt and only horizontal/vertical
        return line.length >= 100 and orientation in ["horizontal", "vertical"]
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
    """Determine if text should be included - prioritize valid dimensions"""
    
    # For all drawing types, only include valid dimension texts
    is_valid = is_valid_dimension(text.text)
    
    if is_valid:
        logger.debug(f"Including valid dimension text: '{text.text}'")
    else:
        logger.debug(f"Excluding invalid dimension text: '{text.text}'")
    
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
        
        logger.info(f"=== FILTERING START (Scale API Compatible) ===")
        logger.info(f"Processing {drawing_type} with {len(regions)} regions")
        logger.info(f"Input: {len(vector_page.lines)} lines, {len(vector_page.texts)} texts")
        logger.info(f"Debug mode: {debug}")
        
        # Apply plattegrond-specific preprocessing
        processed_lines = vector_page.lines
        if drawing_type == "plattegrond":
            logger.info(f"Applying plattegrond-specific preprocessing...")
            
            # Step 1: Filter by length first (≥100pt)
            length_filtered_lines = [line for line in processed_lines if line.length >= 100]
            logger.info(f"After length filter (≥100pt): {len(length_filtered_lines)} lines")
            
            # Step 2: Filter by orientation (only horizontal/vertical)
            orientation_filtered_lines = []
            for line in length_filtered_lines:
                orientation = calculate_orientation(line.p1, line.p2, line.angle)
                if orientation in ["horizontal", "vertical"]:
                    orientation_filtered_lines.append(line)
            logger.info(f"After orientation filter: {len(orientation_filtered_lines)} lines")
            
            # Step 3: Remove duplicate lines
            processed_lines = remove_duplicate_lines(orientation_filtered_lines)
            logger.info(f"After preprocessing: {len(processed_lines)} lines")
        
        region_outputs = []
        total_lines_included = 0
        total_valid_dimension_texts = 0
        
        for region in regions:
            region_lines = []
            region_texts = []
            
            logger.info(f"\nProcessing region: {region.label}")
            
            # Process lines for this region
for line in processed_lines:
    if line_intersects_region(line.p1, line.p2, region.coordinate_block):
        if should_include_line(line, drawing_type, region.label):
            total_lines_included += 1
            filtered_line = FilteredLine(
                length=line.length,
                orientation=calculate_orientation(line.p1, line.p2, line.angle),
                midpoint=calculate_midpoint(line.p1, line.p2)
            )
            region_lines.append(filtered_line)
            
            # Process texts for this region - ONLY VALID DIMENSIONS
            for text in vector_page.texts:
                if text_in_region(text, region.coordinate_block):
                    if should_include_text(text, drawing_type, region.label):
                        total_valid_dimension_texts += 1
                        
                        # Create filtered text with midpoint and orientation
                        filtered_text = FilteredText(
                            text=text.text,
                            midpoint=calculate_text_midpoint(text.bounding_box),
                            orientation=calculate_text_orientation(text.bounding_box)
                        )
                        
                        
                        region_texts.append(filtered_text)
            
            # Create region output
            region_data = RegionData(
                label=region.label,
                lines=region_lines,
                texts=region_texts
            )
            region_outputs.append(region_data)
            
            logger.info(f"  {region.label} results:")
            logger.info(f"    Lines included: {len(region_lines)}")
            logger.info(f"    Valid dimension texts: {len(region_texts)}")
        
        # Create clean output
        output = CleanOutput(
            drawing_type=drawing_type,
            regions=region_outputs
        )
        
        logger.info(f"\n=== FILTERING COMPLETE ===")
        logger.info(f"Total lines included: {total_lines_included}")
        logger.info(f"Total valid dimension texts: {total_valid_dimension_texts}")
        
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
        "service": "Filter API - Scale Compatible",
        "version": "4.0.0-scale-compatible"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Filter API - Scale API Compatible",
        "version": "4.0.0-scale-compatible",
        "description": "Optimized for Scale API with midpoints, orientations, and dimension filtering",
        "scale_api_features": [
            "Text midpoints for precise line-text matching",
            "Text and line orientations for directional matching",
            "Clean production output (minimal JSON)",
            "Valid dimension text filtering only",
            "Debug mode for detailed output"
        ],
        "filtering_features": [
            "Plattegrond-specific preprocessing (≥100pt, horizontal/vertical)",
            "Duplicate line removal",
            "Pure dimension text validation (no letters/symbols)",
            "Regional boundary checking"
        ],
        "endpoints": {
            "/filter/": "Main filtering endpoint (add ?debug=true for detailed output)",
            "/filter-from-vector-api/": "Direct endpoint for Vector Drawing API output",
            "/health": "Health check"
        },
        "output_format": {
            "production": "text, midpoint, orientation only",
            "debug": "includes position and bounding_box fields"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
