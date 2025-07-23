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
    """Check if line intersects with region (no buffer for precision)"""
    try:
        line = LineString([line_p1, line_p2])
        region_box = box(region[0], region[1], region[2], region[3])
        return line.intersects(region_box)
    except Exception:
        # Fallback: check if any endpoint is in region
        x1, y1, x2, y2 = region
        for x, y in [line_p1, line_p2]:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

def text_in_region(text: VectorText, region: List[float]) -> bool:
    """Check if text center is in region"""
    center_x = (text.bounding_box[0] + text.bounding_box[2]) / 2
    center_y = (text.bounding_box[1] + text.bounding_box[3]) / 2
    x1, y1, x2, y2 = region
    return x1 <= center_x <= x2 and y1 <= center_y <= y2

def should_include_line(line: VectorLine, drawing_type: str, region_label: str) -> bool:
    """Determine if line should be included based on drawing type and region"""
    
    # For plattegrond: include ALL lines regardless of length
    if drawing_type == "plattegrond":
        return True
    
    # For other drawing types, apply length filters
    orientation = calculate_orientation(line.p1, line.p2, line.angle)
    
    if drawing_type == "gevelaanzicht":
        return line.length > 40
    elif drawing_type == "detailtekening":
        return line.length > 25
    elif drawing_type == "doorsnede":
        return (line.length > 30 and orientation == "vertical") or line.is_dashed
    elif drawing_type == "bestektekening":
        label_lower = region_label.lower()
        if "grond" in label_lower or "verdieping" in label_lower:
            return True  # Plattegrond rules: all lines
        elif "gevel" in label_lower:
            return line.length > 40
        elif "doorsnede" in label_lower:
            return line.length > 30 and orientation == "vertical"
        else:
            return True  # Default: all lines
    elif drawing_type == "installatietekening":
        return line.stroke_width <= 1 or line.is_dashed
    else:  # unknown
        return line.length > 10

@app.post("/filter/", response_model=CleanOutput)
async def filter_clean(input_data: FilterInput):
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
        
        region_outputs = []
        
        for region in regions:
            region_lines = []
            region_texts = []
            
            logger.info(f"Processing region: {region.label}")
            
            # Process lines for this region
            for line in vector_page.lines:
                # Check if line is in region
                if line_intersects_region(line.p1, line.p2, region.coordinate_block):
                    # Check if line should be included based on rules
                    if should_include_line(line, drawing_type, region.label):
                        clean_line = CleanLine(
                            p1=CleanPoint(x=round(line.p1[0], 1), y=round(line.p1[1], 1)),
                            p2=CleanPoint(x=round(line.p2[0], 1), y=round(line.p2[1], 1)),
                            length=round(line.length, 1),
                            orientation=calculate_orientation(line.p1, line.p2, line.angle),
                            midpoint=calculate_midpoint(line.p1, line.p2)
                        )
                        region_lines.append(clean_line)
            
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
            
            logger.info(f"  {region.label}: {len(region_lines)} lines, {len(region_texts)} texts")
        
        # Create clean output - NO unassigned data, NO metadata
        output = CleanOutput(
            drawing_type=drawing_type,
            regions=region_outputs
        )
        
        total_lines = sum(len(r.lines) for r in region_outputs)
        total_texts = sum(len(r.texts) for r in region_outputs)
        
        logger.info(f"✅ Clean filtering completed:")
        logger.info(f"  Total output: {total_lines} lines, {total_texts} texts")
        logger.info(f"  Regions: {len(region_outputs)}")
        
        return output
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
            "Plattegrond includes ALL lines regardless of length"
        ],
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
