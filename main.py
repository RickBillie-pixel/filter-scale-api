# main.py
import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.box import box
from shapely.affinity import scale as shapely_scale
import math
from redis import Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Filter API",
    description="Filters raw vector JSON data based on drawing type and region context from a vision model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (restricted for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production, e.g., ["https://your-app.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Caching with Redis (configure Redis URL in env for production)
redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

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

class VectorSymbol(BaseModel):
    type: str = Field(..., min_length=1)
    bounding_box: List[float] = Field(..., min_items=4, max_items=4)
    points: List[List[float]] = Field(default_factory=list)

    @validator('bounding_box')
    def validate_bbox(cls, v):
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('Invalid bounding box: x0 < x1 and y0 < y1 required')
        return v

class VectorPage(BaseModel):
    page_size: Dict[str, float] = Field(..., example={"width": 3370.0, "height": 2384.0})
    lines: List[VectorLine] = Field(default_factory=list)
    texts: List[VectorText] = Field(default_factory=list)
    symbols: List[VectorSymbol] = Field(default_factory=list)

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
    regions: List[VisionRegion] = Field(..., min_items=1)
    image_metadata: ImageMetadata

class FilterInput(BaseModel):
    vector_data: VectorData
    vision_output: VisionOutput

class FilteredLine(BaseModel):
    p1: List[float]
    p2: List[float]
    stroke_width: float
    length: float
    color: List[int]
    is_dashed: bool
    angle: Optional[float]
    orientation: str  # "horizontal", "vertical", "diagonal"

class FilteredRegion(BaseModel):
    label: str
    bounding_box: List[float]
    lines: List[FilteredLine]
    texts: List[VectorText]
    symbols: List[VectorSymbol]

class UnassignedElements(BaseModel):
    lines: List[FilteredLine]
    texts: List[VectorText]
    symbols: List[VectorSymbol]

class FilteredData(BaseModel):
    page_number: int
    drawing_type: str
    scale_api_version: str
    regions: List[FilteredRegion]
    unassigned: UnassignedElements

class Metadata(BaseModel):
    processed_elements: int
    filtered_elements: int
    regions_processed: int
    processing_time_seconds: float
    timestamp: str
    config_used: Dict[str, str]

class FilterOutput(BaseModel):
    filtered: FilteredData
    metadata: Metadata
    errors: List[str] = Field(default_factory=list)

# Helper functions
def calculate_orientation(p1: List[float], p2: List[float], angle: Optional[float] = None) -> str:
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

def point_in_region(point: List[float], region: List[float], buffer: float = 0) -> bool:
    x, y = point
    x1, y1, x2, y2 = region
    x1 -= buffer
    y1 -= buffer
    x2 += buffer
    y2 += buffer
    return x1 <= x <= x2 and y1 <= y <= y2

def line_in_region(line_p1: List[float], line_p2: List[float], region: List[float], buffer: float = 0) -> bool:
    line = LineString([line_p1, line_p2])
    region_box = box(region[0] - buffer, region[1] - buffer, region[2] + buffer, region[3] + buffer)
    return line.intersects(region_box)

def text_in_region(text: VectorText, region: List[float], buffer: float = 0) -> bool:
    center_x = (text.bounding_box[0] + text.bounding_box[2]) / 2
    center_y = (text.bounding_box[1] + text.bounding_box[3]) / 2
    return point_in_region([center_x, center_y], region, buffer)

def symbol_in_region(symbol: VectorSymbol, region: List[float], buffer: float = 0) -> bool:
    center_x = (symbol.bounding_box[0] + symbol.bounding_box[2]) / 2
    center_y = (symbol.bounding_box[1] + symbol.bounding_box[3]) / 2
    return point_in_region([center_x, center_y], region, buffer)

def calculate_region_area(region: List[float]) -> float:
    x1, y1, x2, y2 = region
    return abs((x2 - x1) * (y2 - y1))

def process_filter(input_data: FilterInput, debug: bool = False) -> FilterOutput:
    start_time = datetime.now()
    errors = []
    
    try:
        # Validate input
        if not input_data.vector_data.pages:
            errors.append("No pages in vector_data")
            raise HTTPException(status_code=400, detail="No pages in vector_data")
        
        if not input_data.vision_output.regions:
            errors.append("No regions in vision_output")
            raise HTTPException(status_code=400, detail="No regions in vision_output")
        
        # Process first page (extend for multi-page if needed)
        vector_page = input_data.vector_data.pages[0]
        drawing_type = input_data.vision_output.drawing_type
        regions = input_data.vision_output.regions
        
        logger.info(f"Processing page {input_data.vector_data.page_number} of type {drawing_type}")
        
        # Count original elements
        original_count = len(vector_page.lines) + len(vector_page.texts) + len(vector_page.symbols)
        
        filtered_regions = []
        unassigned_lines = []
        unassigned_texts = []
        unassigned_symbols = []
        
        for region in regions:
            region_lines = []
            region_texts = []
            region_symbols = []
            
            # Always include all texts in all regions
            for text in vector_page.texts:
                if text_in_region(text, region.coordinate_block, buffer=15):
                    region_texts.append(text)
            
            # Process lines based on drawing_type
            for line in vector_page.lines:
                # Add orientation (mandatory for all lines)
                line.orientation = calculate_line_orientation(line.p1, line.p2, line.angle)
                
                include = False
                
                if drawing_type == "plattegrond":
                    # Include all lines in plattegrond regions
                    if line_in_region(line.p1, line.p2, region.coordinate_block, buffer=15):
                        include = True
                elif drawing_type == "gevelaanzicht":
                    # Lines > 40pt in gevel regions
                    if line.length > 40 and line_in_region(line.p1, line.p2, region.coordinate_block, buffer=15):
                        include = True
                elif drawing_type == "detailtekening":
                    # Lines > 25pt in detail regions
                    if line.length > 25 and line_in_region(line.p1, line.p2, region.coordinate_block, buffer=15):
                        include = True
                elif drawing_type == "doorsnede":
                    # Vertical lines > 30pt in doorsnede regions
                    if line.length > 30 and line.orientation == "vertical" and line_in_region(line.p1, line.p2, region.coordinate_block, buffer=15):
                        include = True
                elif drawing_type == "bestektekening":
                    # Combined rules
                    if line_in_region(line.p1, line.p2, region.coordinate_block, buffer=0):
                        if "grond" in region.label.lower():
                            include = True  # Plattegrond rules: all lines
                        elif "gevel" in region.label.lower():
                            include = line.length > 40  # Gevel rules
                        elif "doorsnede" in region.label.lower():
                            include = line.length > 30 and line.orientation == "vertical"  # Doorsnede rules
                elif drawing_type == "installatietekening":
                    # Exclude lines >1pt stroke_width unless dashed
                    if line_in_region(line.p1, line.p2, region.coordinate_block, buffer=15):
                        if line.stroke_width <= 1 or line.is_dashed:
                            include = True
                elif drawing_type == "unknown":
                    # Lines > 10pt in largest region
                    if line.length > 10 and line_in_region(line.p1, line.p2, region.coordinate_block, buffer=0):
                        include = True
                
                if include:
                    filtered_line = FilteredLine(
                        p1=line.p1,
                        p2=line.p2,
                        stroke_width=line.stroke_width,
                        length=line.length,
                        color=line.color,
                        is_dashed=line.is_dashed,
                        angle=line.angle,
                        orientation=line.orientation
                    )
                    region_lines.append(filtered_line)
            
            # Include symbols in region
            for symbol in vector_page.symbols:
                if symbol_in_region(symbol, region.coordinate_block, buffer=15):
                    region_symbols.append(symbol)
            
            filtered_regions.append(FilteredRegion(
                label=region.label,
                bounding_box=region.coordinate_block,
                lines=region_lines,
                texts=region_texts,
                symbols=region_symbols
            ))
        
        # Handle unassigned (elements not in any region)
        for line in vector_page.lines:
            if all(not line_in_region(line.p1, line.p2, r.coordinate_block, buffer=15) for r in regions):
                line.orientation = calculate_line_orientation(line.p1, line.p2, line.angle)
                filtered_line = FilteredLine(
                    p1=line.p1,
                    p2=line.p2,
                    stroke_width=line.stroke_width,
                    length=line.length,
                    color=line.color,
                    is_dashed=line.is_dashed,
                    angle=line.angle,
                    orientation=line.orientation
                )
                unassigned_lines.append(filtered_line)
        
        for text in vector_page.texts:
            if all(not text_in_region(text, r.coordinate_block, buffer=15) for r in regions):
                unassigned_texts.append(text)
        
        for symbol in vector_page.symbols:
            if all(not symbol_in_region(symbol, r.coordinate_block, buffer=15) for r in regions):
                unassigned_symbols.append(symbol)
        
        filtered_data = FilteredData(
            page_number=input_data.vector_data.page_number,
            drawing_type=drawing_type,
            scale_api_version=input_data.vision_output.scale_api_version,
            regions=filtered_regions,
            unassigned=UnassignedElements(
                lines=unassigned_lines,
                texts=unassigned_texts,
                symbols=unassigned_symbols
            )
        )
        
        # Count filtered elements
        filtered_count = sum(
            len(r.lines) + len(r.texts) + len(r.symbols)
            for r in filtered_regions
        ) + len(unassigned_lines) + len(unassigned_texts) + len(unassigned_symbols)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metadata = Metadata(
            processed_elements=original_count,
            filtered_elements=filtered_count,
            regions_processed=len(regions),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            config_used={
                "scale_api_version": input_data.vision_output.scale_api_version
            }
        )
        
        logger.info(f"Processed {original_count} elements, filtered {filtered_count} elements")
        
        response = FilterOutput(
            filtered=filtered_data,
            metadata=metadata,
            errors=errors
        )
        
        if debug:
            response["debug"] = {"raw_vector_data": vector_page.dict()}
        
        return response
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return FilterOutput(
            filtered=None,
            metadata=Metadata(
                processed_elements=0,
                filtered_elements=0,
                regions_processed=0,
                processing_time_seconds=0,
                timestamp=datetime.now().isoformat(),
                config_used={}
            ),
            errors=[str(e)]
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Filter API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)