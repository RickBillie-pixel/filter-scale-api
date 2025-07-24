import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import LineString, box
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clean Filter API - Compatible with Vector Drawing API",
    description="Returns clean, focused output per region - works with Vector Drawing API format",
    version="3.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Input Models - Compatible with Vector Drawing API
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

# Clean Output Models - Only Essential Data
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
    bounding_box: List[float]  # [x1, y1, x2, y2]

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
        
        if dx < 1:  # Practically vertical
            return "vertical"
        elif dy < 1:  # Practically horizontal
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

def line_intersects_region(line_p1: List[float], line_p2: List[float], region: List[float]) -> bool:
    """Check if line intersects with region - improved with debugging and multiple methods"""
    x1, y1, x2, y2 = region
    
    # Method 1: Check if any endpoint is in region
    for x, y in [line_p1, line_p2]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    
    # Method 2: Check if line completely spans the region in x or y direction
    line_x1, line_y1 = line_p1
    line_x2, line_y2 = line_p2
    
    # Check if line completely spans the region in x or y direction
    if ((line_x1 <= x1 and line_x2 >= x2) or (line_x1 >= x2 and line_x2 <= x1)) and \
       ((min(line_y1, line_y2) <= y2) and (max(line_y1, line_y2) >= y1)):
        return True
        
    if ((line_y1 <= y1 and line_y2 >= y2) or (line_y1 >= y2 and line_y2 <= y1)) and \
       ((min(line_x1, line_x2) <= x2) and (max(line_x1, line_x2) >= x1)):
        return True
    
    # Method 3: Use Shapely as fallback (with error handling)
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
        return line.length > 50  # ✅ CORRECT: Plattegrond lines > 50pt
    elif drawing_type == "gevelaanzicht":
        return line.length > 40  # Gevelaanzicht: lines > 40pt
    elif drawing_type == "detailtekening":
        return line.length > 25  # Detailtekening: lines > 25pt
    elif drawing_type == "doorsnede":
        return (line.length > 30 and orientation == "vertical") or line.is_dashed  # Doorsnede: vertical > 30pt OR dashed
    elif drawing_type == "bestektekening":
        label_lower = region_label.lower()
        if "grond" in label_lower or "verdieping" in label_lower:
            return line.length > 50  # Plattegrond rules: lines > 50pt
        elif "gevel" in label_lower:
            return line.length > 40  # Gevel rules: lines > 40pt
        elif "doorsnede" in label_lower:
            return line.length > 30 and orientation == "vertical"  # Doorsnede rules: vertical > 30pt
        else:
            return line.length > 25  # Default: lines > 25pt
    elif drawing_type == "installatietekening":
        return line.stroke_width <= 1 or line.is_dashed  # Installatie: thin lines OR dashed
    else:  # unknown
        return line.length > 10  # Unknown: lines > 10pt

def convert_vector_drawing_api_format(raw_vector_data: Dict) -> VectorData:
    """Convert Vector Drawing API format to our internal format"""
    try:
        logger.info("=== Converting Vector Drawing API format ===")
        
        pages = raw_vector_data.get("pages", [])
        if not pages:
            raise ValueError("No pages found in vector data")
        
        converted_pages = []
        
        for page_data in pages:
            page_size = page_data.get("page_size", {"width": 595.0, "height": 842.0})
            
            # Extract texts - direct format
            texts = []
            for text_data in page_data.get("texts", []):
                position = text_data.get("position", {"x": 0, "y": 0})
                bbox = text_data.get("bbox", {})
                
                # Convert position to [x, y] format
                if isinstance(position, dict):
                    pos_list = [float(position.get("x", 0)), float(position.get("y", 0))]
                else:
                    pos_list = [float(position[0]), float(position[1])]
                
                # Convert bbox to [x1, y1, x2, y2] format  
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
            
            # Extract lines from drawings
            lines = []
            drawings = page_data.get("drawings", {})
            
            logger.info(f"Drawings structure: {list(drawings.keys())}")
            
            for line_data in drawings.get("lines", []):
                logger.info(f"Processing line: {line_data}")
                
                # Handle Vector Drawing API format: {"type": "line", "p1": {"x": 100, "y": 200}, "p2": {"x": 300, "y": 400}, "length": 250}
                p1 = line_data.get("p1", {"x": 0, "y": 0})
                p2 = line_data.get("p2", {"x": 0, "y": 0})
                
                # Convert to [x, y] format
                if isinstance(p1, dict):
                    p1_list = [float(p1.get("x", 0)), float(p1.get("y", 0))]
                else:
                    p1_list = [float(p1[0]), float(p1[1])]
                
                if isinstance(p2, dict):
                    p2_list = [float(p2.get("x", 0)), float(p2.get("y", 0))]
                else:
                    p2_list = [float(p2[0]), float(p2[1])]
                
                # Get other properties
                length = float(line_data.get("length", 0))
                width = float(line_data.get("width", 1.0))
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
                
                logger.info(f"Converted line: {p1_list} -> {p2_list}, length: {length}")
            
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
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting Vector Drawing API format: {e}")
        raise ValueError(f"Failed to convert Vector Drawing API format: {str(e)}")

@app.post("/filter/", response_model=CleanOutput)
async def filter_clean(input_data: FilterInput, debug: bool = Query(False)):
    """Filter data and return clean, focused output per region"""
    
    try:
        # Get first page data
        if not input_data.vector_data.pages:
            raise HTTPException(status_code=400, detail="No pages in vector_data")
        
        vector_page = input_data.vector_data.pages[0]
        drawing_type = input_data.vision_output.drawing_type
        regions = input_data.vision_output.regions
        
        logger.info(f"Processing {drawing_type} with {len(regions)} regions")
        logger.info(f"Input: {len(vector_page.lines)} lines, {len(vector_page.texts)} texts")
        
        # Log first few lines for debugging
        if debug and vector_page.lines:
            logger.info("Sample lines:")
            for i, line in enumerate(vector_page.lines[:3]):
                logger.info(f"  Line {i}: p1={line.p1}, p2={line.p2}, length={line.length}")
        
        region_outputs = []
        total_lines_processed = 0
        
        for region in regions:
            region_lines = []
            region_texts = []
            lines_in_region = 0
            
            logger.info(f"Processing region: {region.label}")
            logger.info(f"  Region bounds: {region.coordinate_block}")
            
            # Process lines for this region
            for i, line in enumerate(vector_page.lines):
                # Check if line is in region
                is_in_region = line_intersects_region(line.p1, line.p2, region.coordinate_block)
                
                if debug and i < 5:  # Debug first 5 lines
                    logger.info(f"  Line {i}: {line.p1} -> {line.p2}, in_region: {is_in_region}")
                
                if is_in_region:
                    lines_in_region += 1
                    
                    # Check if line should be included based on rules
                    should_include = should_include_line(line, drawing_type, region.label)
                    
                    if debug and lines_in_region <= 3:
                        logger.info(f"    Should include: {should_include} (type: {drawing_type}, length: {line.length})")
                    
                    if should_include:
                        clean_line = CleanLine(
                            p1=CleanPoint(x=round(line.p1[0], 1), y=round(line.p1[1], 1)),
                            p2=CleanPoint(x=round(line.p2[0], 1), y=round(line.p2[1], 1)),
                            length=round(line.length, 1),
                            orientation=calculate_orientation(line.p1, line.p2, line.angle),
                            midpoint=calculate_midpoint(line.p1, line.p2)
                        )
                        region_lines.append(clean_line)
            
            total_lines_processed += lines_in_region
            
            # Process texts for this region
            for text in vector_page.texts:
                if text_in_region(text, region.coordinate_block):
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
            
            logger.info(f"  {region.label}: {len(region_lines)} lines (from {lines_in_region} in region), {len(region_texts)} texts")
        
        # Create clean output - NO unassigned data, NO metadata
        output = CleanOutput(
            drawing_type=drawing_type,
            regions=region_outputs
        )
        
        total_lines = sum(len(r.lines) for r in region_outputs)
        total_texts = sum(len(r.texts) for r in region_outputs)
        
        logger.info(f"✅ Clean filtering completed:")
        logger.info(f"  Total lines found in regions: {total_lines_processed}")
        logger.info(f"  Total lines included: {total_lines}")
        logger.info(f"  Total texts: {total_texts}")
        logger.info(f"  Regions: {len(region_outputs)}")
        
        if total_lines == 0:
            logger.warning("⚠️  NO LINES FOUND! This suggests a coordinate system or intersection problem.")
        
        return output
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
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
        "service": "Clean Filter API - Vector Drawing API Compatible",
        "version": "3.1.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Clean Filter API - Vector Drawing API Compatible",
        "version": "3.1.0",
        "description": "Returns clean, focused output per region - works with Vector Drawing API format",
        "features": [
            "Compatible with Vector Drawing API format",
            "Clean output per region only",
            "No unassigned data", 
            "Correct length filtering per drawing type",
            "Processes p1/p2 as {x,y} objects or [x,y] arrays"
        ],
        "drawing_types": {
            "plattegrond": "Lines > 50pt",
            "gevelaanzicht": "Lines > 40pt",
            "detailtekening": "Lines > 25pt", 
            "doorsnede": "Vertical lines > 30pt OR dashed lines",
            "bestektekening": "Region-specific rules",
            "installatietekening": "Lines ≤ 1pt stroke OR dashed",
            "unknown": "Lines > 10pt"
        },
        "endpoints": {
            "/filter/": "Main filtering endpoint (requires converted format)",
            "/filter-from-vector-api/": "Direct endpoint for Vector Drawing API output",
            "/health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
