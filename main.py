import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from shapely.geometry import LineString, box
import math
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Filter API",
    description="Filters lines per region based on drawing type, returns minified output for Scale API",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class VectorLine(BaseModel):
    p1: List[float] = Field(..., min_items=2, max_items=2)
    p2: List[float] = Field(..., min_items=2, max_items=2)
    stroke_width: float = Field(..., ge=0.0)
    length: float = Field(..., ge=0.0)
    color: List[int] = Field(..., min_items=3, max_items=3)
    is_dashed: bool = Field(default=False)
    angle: Optional[float] = Field(default=None, ge=0.0, le=360.0)

class VectorText(BaseModel):
    text: str = Field(..., min_length=1)
    position: List[float] = Field(..., min_items=2, max_items=2)
    font_size: Optional[float] = Field(default=None, ge=0.0)
    bounding_box: List[float] = Field(..., min_items=4, max_items=4)

    @validator('bounding_box')
    def validate_bbox(cls, v):
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('Invalid bounding box: x0 < x1 and y0 < y1 required')
        return v

class VectorPage(BaseModel):
    page_size: Dict[str, float] = Field(..., example={"width": 3370.0, "height": 2384.0})
    lines: List[VectorLine] = Field(default_factory=list)
    texts: List[VectorText] = Field(default_factory=list)

class VectorData(BaseModel):
    page_number: int = Field(..., ge=1)
    pages: List[VectorPage] = Field(..., min_items=1)

class VisionRegion(BaseModel):
    label: str = Field(..., min_length=1)
    coordinate_block: List[float] = Field(..., min_items=4, max_items=4)

    @validator('coordinate_block')
    def validate_coord_block(cls, v):
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('Invalid coordinate block: x0 < x1 and y0 < y1 required')
        return v

class ImageMetadata(BaseModel):
    image_width_pixels: int = Field(..., ge=1)
    image_height_pixels: int = Field(..., ge=1)
    image_dpi_x: Optional[float] = Field(default=None, ge=1.0)
    image_dpi_y: Optional[float] = Field(default=None, ge=1.0)

class VisionOutput(BaseModel):
    drawing_type: str = Field(..., pattern="^(plattegrond|gevelaanzicht|detailtekening|doorsnede|bestektekening|installatietekening|unknown)$")
    scale_api_version: str = Field(..., min_length=1)
    regions: List[VisionRegion] = Field(..., min_items=0)
    image_metadata: ImageMetadata

class FilterInput(BaseModel):
    vector_data: VectorData
    vision_output: VisionOutput

# Minified output models for Scale API compatibility
class ScaleApiPoint(BaseModel):
    x: float
    y: float

class ScaleApiLine(BaseModel):
    type: str = "line"
    p1: ScaleApiPoint
    p2: ScaleApiPoint
    length: float
    orientation: str
    midpoint: ScaleApiPoint

class ScaleApiText(BaseModel):
    text: str
    position: ScaleApiPoint

class RegionOutput(BaseModel):
    label: str
    vector_data: List[ScaleApiLine]
    texts: List[ScaleApiText]

class MinifiedOutput(BaseModel):
    drawing_type: str
    page_number: int
    regions: List[RegionOutput]
    unassigned_vector_data: List[ScaleApiLine]
    unassigned_texts: List[ScaleApiText]
    metadata: Dict[str, Any]

# Helper functions
def calculate_orientation(p1: List[float], p2: List[float], angle: Optional[float] = None) -> str:
    """Calculate line orientation"""
    if angle is not None:
        normalized_angle = abs(angle % 180)
        if normalized_angle < 10 or normalized_angle > 170:
            return "horizontal"
        elif 80 < normalized_angle < 100:
            return "vertical"
        else:
            return "diagonal"
    else:
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        if dx == 0:
            return "vertical"
        elif dy == 0:
            return "horizontal"
        else:
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 10 or angle_deg > 170:
                return "horizontal"
            elif 80 < angle_deg < 100:
                return "vertical"
            else:
                return "diagonal"

def calculate_midpoint(p1: List[float], p2: List[float]) -> ScaleApiPoint:
    """Calculate midpoint of a line"""
    return ScaleApiPoint(
        x=round((p1[0] + p2[0]) / 2, 2),
        y=round((p1[1] + p2[1]) / 2, 2)
    )

def line_in_region(line_p1: List[float], line_p2: List[float], region: List[float], buffer: float = 0) -> bool:
    """Check if line intersects with region"""
    line = LineString([line_p1, line_p2])
    region_box = box(region[0] - buffer, region[1] - buffer, region[2] + buffer, region[3] + buffer)
    return line.intersects(region_box)

def text_in_region(text: VectorText, region: List[float], buffer: float = 0) -> bool:
    """Check if text is in region"""
    center_x = (text.bounding_box[0] + text.bounding_box[2]) / 2
    center_y = (text.bounding_box[1] + text.bounding_box[3]) / 2
    return (center_x >= region[0] - buffer and center_x <= region[2] + buffer and
            center_y >= region[1] - buffer and center_y <= region[3] + buffer)

def calculate_region_area(region: List[float]) -> float:
    """Calculate region area"""
    x1, y1, x2, y2 = region
    return abs((x2 - x1) * (y2 - y1))

def convert_line_to_scale_format(line: VectorLine) -> ScaleApiLine:
    """Convert line to Scale API format with midpoint"""
    orientation = calculate_orientation(line.p1, line.p2, line.angle)
    midpoint = calculate_midpoint(line.p1, line.p2)
    
    return ScaleApiLine(
        type="line",
        p1=ScaleApiPoint(x=round(line.p1[0], 2), y=round(line.p1[1], 2)),
        p2=ScaleApiPoint(x=round(line.p2[0], 2), y=round(line.p2[1], 2)),
        length=round(line.length, 2),
        orientation=orientation,
        midpoint=midpoint
    )

def convert_text_to_scale_format(text: VectorText) -> ScaleApiText:
    """Convert text to Scale API format"""
    return ScaleApiText(
        text=text.text,
        position=ScaleApiPoint(x=round(text.position[0], 2), y=round(text.position[1], 2))
    )

@app.post("/filter/", response_model=MinifiedOutput)
@limiter.limit("10/minute")
async def filter_data(request: Request, input_data: FilterInput, debug: bool = Query(False)):
    """Filter lines per region and return minified output for Scale API"""
    start_time = datetime.now()
    
    try:
        # Validate input
        if not input_data.vector_data.pages:
            raise HTTPException(status_code=400, detail="No pages in vector_data")
        
        if input_data.vision_output.drawing_type in ["detailtekening", "unknown"] and not input_data.vision_output.regions:
            raise HTTPException(status_code=400, detail=f"No regions provided for {input_data.vision_output.drawing_type}")
        
        vector_page = input_data.vector_data.pages[0]
        drawing_type = input_data.vision_output.drawing_type
        regions = input_data.vision_output.regions
        
        logger.info(f"Processing page {input_data.vector_data.page_number} of type {drawing_type}")
        logger.info(f"Input: {len(vector_page.lines)} lines, {len(vector_page.texts)} texts")
        
        original_count = len(vector_page.lines) + len(vector_page.texts)
        
        # Process regions
        region_outputs = []
        unassigned_texts = vector_page.texts.copy()
        processed_lines = set()
        
        # For unknown drawing type, select largest region
        if drawing_type == "unknown" and regions:
            regions = [max(regions, key=lambda r: calculate_region_area(r.coordinate_block))]
        
        for region in regions:
            region_lines = []
            region_texts = []
            
            # Process texts for this region
            texts_to_remove = []
            for text in unassigned_texts:
                if text_in_region(text, region.coordinate_block, buffer=15 if drawing_type != "bestektekening" else 0):
                    region_texts.append(convert_text_to_scale_format(text))
                    texts_to_remove.append(text)
            
            # Remove assigned texts from unassigned list
            for text in texts_to_remove:
                unassigned_texts.remove(text)
            
            # Process lines for this region based on drawing type rules
            for i, line in enumerate(vector_page.lines):
                if i in processed_lines:
                    continue
                    
                orientation = calculate_orientation(line.p1, line.p2, line.angle)
                include = False
                buffer = 15 if drawing_type != "bestektekening" else 0
                
                if line_in_region(line.p1, line.p2, region.coordinate_block, buffer=buffer):
                    # Apply drawing type specific filtering rules
                    if drawing_type == "plattegrond":
                        include = True  # All lines
                    elif drawing_type == "gevelaanzicht":
                        include = line.length > 40  # Lines > 40pt
                    elif drawing_type == "detailtekening":
                        include = line.length > 25  # Lines > 25pt
                    elif drawing_type == "doorsnede":
                        # Vertical lines > 30pt OR dashed lines
                        include = (line.length > 30 and orientation == "vertical") or line.is_dashed
                    elif drawing_type == "bestektekening":
                        # Apply region-specific rules for bestektekening
                        label_lower = region.label.lower()
                        if "grond" in label_lower:
                            include = True  # Plattegrond rules: all lines
                        elif "gevel" in label_lower:
                            include = line.length > 40  # Gevel rules
                        elif "doorsnede" in label_lower:
                            include = line.length > 30 and orientation == "vertical"  # Doorsnede rules
                        else:
                            include = True  # Default: all lines
                    elif drawing_type == "installatietekening":
                        # Lines ≤ 1pt stroke OR dashed lines
                        include = line.stroke_width <= 1 or line.is_dashed
                    elif drawing_type == "unknown":
                        include = line.length > 10  # Lines > 10pt
                
                if include:
                    region_lines.append(convert_line_to_scale_format(line))
                    processed_lines.add(i)
            
            # Create region output
            region_output = RegionOutput(
                label=region.label,
                vector_data=region_lines,
                texts=region_texts
            )
            region_outputs.append(region_output)
            
            logger.info(f"Region '{region.label}': {len(region_lines)} lines, {len(region_texts)} texts")
        
        # Process unassigned lines
        unassigned_lines = []
        for i, line in enumerate(vector_page.lines):
            if i not in processed_lines:
                unassigned_lines.append(convert_line_to_scale_format(line))
        
        # Convert unassigned texts
        unassigned_texts_converted = [convert_text_to_scale_format(text) for text in unassigned_texts]
        
        # Calculate processing stats
        total_filtered_lines = sum(len(r.vector_data) for r in region_outputs) + len(unassigned_lines)
        total_filtered_texts = sum(len(r.texts) for r in region_outputs) + len(unassigned_texts_converted)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create minified output
        output = MinifiedOutput(
            drawing_type=drawing_type,
            page_number=input_data.vector_data.page_number,
            regions=region_outputs,
            unassigned_vector_data=unassigned_lines,
            unassigned_texts=unassigned_texts_converted,
            metadata={
                "processed_elements": original_count,
                "filtered_lines": total_filtered_lines,
                "filtered_texts": total_filtered_texts,
                "regions_processed": len(regions),
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "scale_api_version": input_data.vision_output.scale_api_version
            }
        )
        
        logger.info(f"✅ Filtering completed:")
        logger.info(f"  Original: {original_count} elements")
        logger.info(f"  Filtered: {total_filtered_lines} lines, {total_filtered_texts} texts")
        logger.info(f"  Regions: {len(region_outputs)}")
        logger.info(f"  Processing time: {processing_time:.3f}s")
        
        return output
    
    except HTTPException as e:
        raise e
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Filter API",
        "version": "2.1.0",
        "description": "Minified output for Scale API compatibility",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Filter API",
        "version": "2.1.0",
        "description": "Filters lines per region based on drawing type, returns minified output for Scale API",
        "features": [
            "Region-based line filtering with drawing type rules",
            "Minified output format for Scale API",
            "Line midpoint calculation",
            "Optimized text processing",
            "No symbols (skipped for performance)"
        ],
        "drawing_types": {
            "plattegrond": "All lines included",
            "gevelaanzicht": "Lines > 40pt",
            "detailtekening": "Lines > 25pt", 
            "doorsnede": "Vertical lines > 30pt OR dashed lines",
            "bestektekening": "Region-specific rules (grond=all, gevel>40pt, doorsnede=vertical>30pt)",
            "installatietekening": "Lines ≤ 1pt stroke OR dashed",
            "unknown": "Lines > 10pt in largest region"
        },
        "output_format": "Minified for Scale API compatibility with midpoints"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
