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
    title="Clean Filter API",
    description="Returns clean, focused output per region - only essential data",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Input Models
class VectorLine(BaseModel):
    p1: List[float]
    p2: List[float] 
    stroke_width: float
    length: float
    color: List[int]
    is_dashed: bool = False
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
    
    # Method 2: Check if line crosses region boundaries
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
        return line.length > 50  # Plattegrond: lines > 50pt
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
            logger.warning(f"Sample region bounds: {regions[0].coordinate_block if regions else 'None'}")
            if vector_page.lines:
                sample_line = vector_page.lines[0]
                logger.warning(f"Sample line: {sample_line.p1} -> {sample_line.p2}")
        
        return output
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/debug/", response_model=Dict[str, Any])
async def debug_filter(input_data: FilterInput):
    """Debug endpoint to diagnose line filtering issues"""
    
    try:
        vector_page = input_data.vector_data.pages[0]
        regions = input_data.vision_output.regions
        
        debug_info = {
            "total_lines": len(vector_page.lines),
            "total_texts": len(vector_page.texts),
            "drawing_type": input_data.vision_output.drawing_type,
            "regions": [],
            "sample_lines": [],
            "coordinate_analysis": {}
        }
        
        # Sample first 5 lines
        for i, line in enumerate(vector_page.lines[:5]):
            debug_info["sample_lines"].append({
                "index": i,
                "p1": line.p1,
                "p2": line.p2,
                "length": line.length,
                "stroke_width": line.stroke_width
            })
        
        # Analyze coordinate ranges
        if vector_page.lines:
            all_x = []
            all_y = []
            for line in vector_page.lines:
                all_x.extend([line.p1[0], line.p2[0]])
                all_y.extend([line.p1[1], line.p2[1]])
            
            debug_info["coordinate_analysis"] = {
                "x_range": [min(all_x), max(all_x)],
                "y_range": [min(all_y), max(all_y)],
                "total_points": len(all_x)
            }
        
        # Test each region
        for region in regions:
            region_debug = {
                "label": region.label,
                "bounds": region.coordinate_block,
                "lines_in_region": 0,
                "lines_included": 0,
                "texts_in_region": 0,
                "sample_intersections": []
            }
            
            # Test line intersections
            for i, line in enumerate(vector_page.lines[:10]):  # First 10 lines
                is_in_region = line_intersects_region(line.p1, line.p2, region.coordinate_block)
                if is_in_region:
                    region_debug["lines_in_region"] += 1
                    should_include = should_include_line(line, input_data.vision_output.drawing_type, region.label)
                    if should_include:
                        region_debug["lines_included"] += 1
                
                if i < 5:  # Sample first 5
                    region_debug["sample_intersections"].append({
                        "line_index": i,
                        "p1": line.p1,
                        "p2": line.p2,
                        "in_region": is_in_region,
                        "would_include": should_include_line(line, input_data.vision_output.drawing_type, region.label) if is_in_region else False
                    })
            
            # Count texts in region
            for text in vector_page.texts:
                if text_in_region(text, region.coordinate_block):
                    region_debug["texts_in_region"] += 1
            
            debug_info["regions"].append(region_debug)
        
        return debug_info
    
    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Clean Filter API",
        "version": "3.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Clean Filter API",
        "version": "3.0.0",
        "description": "Returns clean, focused output per region - only essential data",
        "features": [
            "Clean output per region only",
            "No unassigned data", 
            "No unnecessary metadata",
            "Precise line filtering with orientation and midpoint",
            "Text with bounding box preserved",
            "Correct length filtering per drawing type",
            "Debug endpoint for troubleshooting"
        ],
        "drawing_types": {
            "plattegrond": "Lines > 50pt",
            "gevelaanzicht": "Lines > 40pt",
            "detailtekening": "Lines > 25pt", 
            "doorsnede": "Vertical lines > 30pt OR dashed lines",
            "bestektekening": "Region-specific rules (grond>50pt, gevel>40pt, doorsnede=vertical>30pt, default>25pt)",
            "installatietekening": "Lines ≤ 1pt stroke OR dashed",
            "unknown": "Lines > 10pt"
        },
        "endpoints": {
            "/filter/": "Main filtering endpoint",
            "/debug/": "Debug line intersection issues",
            "/filter/?debug=true": "Filter with debug logging"
        },
        "output_structure": {
            "drawing_type": "string",
            "regions": [
                {
                    "label": "region name",
                    "lines": [
                        {
                            "p1": {"x": float, "y": float},
                            "p2": {"x": float, "y": float},
                            "length": float,
                            "orientation": "horizontal|vertical|diagonal",
                            "midpoint": {"x": float, "y": float}
                        }
                    ],
                    "texts": [
                        {
                            "text": "string",
                            "position": {"x": float, "y": float},
                            "bounding_box": [x1, y1, x2, y2]
                        }
                    ]
                }
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
