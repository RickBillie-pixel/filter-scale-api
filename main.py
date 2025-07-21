from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple
import math
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pre-Filter API",
    description="Filters vector data and texts for scale processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

class Point(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

class BBox(BaseModel):
    x0: float = Field(..., description="Left boundary")
    y0: float = Field(..., description="Top boundary")
    x1: float = Field(..., description="Right boundary")
    y1: float = Field(..., description="Bottom boundary")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

class Text(BaseModel):
    text: str = Field(..., min_length=1, description="Text content")
    position: Dict[str, float] = Field(..., description="Text position coordinates")
    bbox: Optional[BBox] = Field(None, description="Bounding box of text")
    source: Optional[str] = Field(None, description="Source identifier")

    @validator('position')
    def validate_position(cls, v):
        required_keys = {'x', 'y'}
        if not required_keys.issubset(v.keys()):
            raise ValueError('Position must contain x and y coordinates')
        return v

class Line(BaseModel):
    p1: Dict[str, float] = Field(..., description="Start point coordinates")
    p2: Dict[str, float] = Field(..., description="End point coordinates")
    length: float = Field(..., gt=0, description="Length of the line")
    width: Optional[float] = Field(None, description="Line width/thickness")

    @validator('p1', 'p2')
    def validate_points(cls, v):
        required_keys = {'x', 'y'}
        if not required_keys.issubset(v.keys()):
            raise ValueError('Points must contain x and y coordinates')
        return v

class Page(BaseModel):
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    texts: List[Text] = Field(default_factory=list, description="Text elements on page")
    lines: List[Line] = Field(default_factory=list, description="Line elements on page (legacy format)")
    drawings: Optional[Dict[str, List[Dict[str, Any]]]] = Field(None, description="Vector drawings from Vector Drawing API")

    def get_all_lines(self) -> List[Line]:
        """Extract lines from both legacy format and new drawings format"""
        all_lines = []
        all_lines.extend(self.lines)
        if self.drawings and "lines" in self.drawings:
            for line_data in self.drawings["lines"]:
                try:
                    line = Line(p1=line_data["p1"], p2=line_data["p2"], length=line_data["length"], width=line_data.get("width"))
                    all_lines.append(line)
                except Exception as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
        return all_lines

class FilterConfig(BaseModel):
    min_line_length: float = Field(45.0, gt=0, description="Minimum line length to include")
    keep_all_text: bool = Field(True, description="Keep all text types (not just numeric)")
    include_diagonal_lines: bool = Field(True, description="Whether to include diagonal lines")
    diagonal_tolerance: float = Field(0.1, ge=0, le=1, description="Tolerance for diagonal detection")

class InputData(BaseModel):
    pages: List[Page] = Field(..., min_items=1, description="List of pages to process")
    config: Optional[FilterConfig] = Field(default_factory=FilterConfig, description="Filtering configuration")

class ProcessedLine(BaseModel):
    p1: Dict[str, float]
    p2: Dict[str, float]
    length: float
    width: Optional[float]
    orientation: str
    midpoint: Tuple[float, float]

class ProcessedText(BaseModel):
    text: str
    position: Dict[str, float]
    bbox: Optional[BBox]
    source: Optional[str]
    orientation: str
    numeric_value: Optional[float]
    midpoint: Tuple[float, float]

class ProcessedPage(BaseModel):
    page_number: int
    texts: List[ProcessedText]
    lines: List[ProcessedLine]
    stats: Dict[str, Any]

class OutputData(BaseModel):
    pages: List[ProcessedPage]
    processing_stats: Dict[str, Any]

# Compiled regex for better performance
NUMERIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?)')

@lru_cache(maxsize=1000)
def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text, handling suffixes like '+p'."""
    match = NUMERIC_PATTERN.search(text)
    return float(match.group(1)) if match else None

def get_midpoint(p1: Dict[str, float], p2: Dict[str, float]) -> Tuple[float, float]:
    """Calculate midpoint between two points."""
    return ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2)

def get_text_midpoint(text: Text) -> Tuple[float, float]:
    """Get text midpoint from position or bbox if available."""
    if text.bbox:
        return ((text.bbox.x0 + text.bbox.x1) / 2, (text.bbox.y0 + text.bbox.y1) / 2)
    return (text.position['x'], text.position['y'])

def assign_orientation(line: Line, tolerance: float = 0.1) -> str:
    """Assign orientation to line based on slope."""
    dx = abs(line.p2['x'] - line.p1['x'])
    dy = abs(line.p2['y'] - line.p1['y'])
    if dy / max(1, dx) < tolerance:
        return "horizontal"
    elif dx / max(1, dy) < tolerance:
        return "vertical"
    return "diagonal"

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def assign_text_orientation(text: Text, lines: List[Line], tolerance: float = 300) -> str:
    """Assign orientation to text based on nearest line or bbox aspect ratio."""
    midpoint = get_text_midpoint(text)
    nearest_distance = float('inf')
    nearest_orientation = None
    for line in lines:
        line_midpoint = get_midpoint(line.p1, line.p2)
        distance = calculate_distance(midpoint, line_midpoint)
        if distance < nearest_distance and distance < tolerance:
            nearest_distance = distance
            nearest_orientation = assign_orientation(line)
    if nearest_orientation and nearest_orientation in ["horizontal", "vertical"]:
        return nearest_orientation
    if text.bbox:
        width, height = text.bbox.width, text.bbox.height
        if width > height and height < 15:
            return "horizontal"
        elif height > width and width < 15:
            return "vertical"
    return "unknown"

def process_page_data(page: Page, config: FilterConfig) -> ProcessedPage:
    """Process a single page - CPU intensive operation."""
    # Text Processing
    processed_texts = []
    for text in page.texts:
        midpoint = get_text_midpoint(text)
        numeric_value = extract_numeric_value(text.text)
        orientation = assign_text_orientation(text, page.get_all_lines(), 300)
        processed_texts.append(ProcessedText(
            text=text.text,
            position=text.position,
            bbox=text.bbox,
            source=text.source,
            orientation=orientation,
            numeric_value=numeric_value,
            midpoint=midpoint
        ))
    logger.info(f"Processed all {len(processed_texts)} texts with center points and orientations")

    # Line Processing
    all_lines = page.get_all_lines()
    original_line_count = len(all_lines)
    filtered_lines = []
    for line in all_lines:
        if line.length >= config.min_line_length:
            orientation = assign_orientation(line, config.diagonal_tolerance)
            if config.include_diagonal_lines or orientation != "diagonal":
                filtered_lines.append(ProcessedLine(
                    p1=line.p1,
                    p2=line.p2,
                    length=line.length,
                    width=line.width,
                    orientation=orientation,
                    midpoint=get_midpoint(line.p1, line.p2)
                ))
    logger.info(f"Filtered to {len(filtered_lines)} lines (min length: {config.min_line_length})")

    # Track highest dimensions
    highest_dimensions = {
        "horizontal": max(
            [t for t in processed_texts if t.orientation == "horizontal" and t.numeric_value is not None],
            key=lambda x: x.numeric_value,
            default=ProcessedText(text="", position={"x": 0, "y": 0}, bbox=None, source=None, orientation="horizontal", numeric_value=0, midpoint=(0, 0))
        ),
        "vertical": max(
            [t for t in processed_texts if t.orientation == "vertical" and t.numeric_value is not None],
            key=lambda x: x.numeric_value,
            default=ProcessedText(text="", position={"x": 0, "y": 0}, bbox=None, source=None, orientation="vertical", numeric_value=0, midpoint=(0, 0))
        )
    }

    return ProcessedPage(
        page_number=page.page_number,
        texts=processed_texts,
        lines=filtered_lines,
        stats={
            "original_texts": len(page.texts),
            "filtered_texts": len(processed_texts),
            "original_lines": original_line_count,
            "filtered_lines": len(filtered_lines),
            "deduplication": "disabled",
            "filter_config_used": {
                "min_line_length": config.min_line_length,
                "keep_all_text": config.keep_all_text,
                "include_diagonal_lines": config.include_diagonal_lines
            },
            "highest_dimensions": {
                "horizontal": {"text": highest_dimensions["horizontal"].text, "numeric_value": highest_dimensions["horizontal"].numeric_value},
                "vertical": {"text": highest_dimensions["vertical"].text, "numeric_value": highest_dimensions["vertical"].numeric_value}
            }
        }
    )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Pre-Filter API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {"status": "healthy", "service": "Pre-Filter API", "version": "1.0.0"}

@app.post("/pre-scale", response_model=OutputData)
@app.post("/pre-filter/", response_model=OutputData)
async def pre_filter(data: InputData):
    """
    Filter vector data and texts for scale processing.
    
    **New filtering behavior:**
    - **Texts**: Keeps ALL text types by default, calculates center points, extracts numeric values, assigns orientations
    - **Lines**: Filters on length >= 45, determines orientations, calculates midpoints
    - **No deduplication**: All qualifying lines are kept
    
    - **pages**: List of pages containing texts and lines
    - **config**: Optional filtering configuration
        - min_line_length: 45.0 (default)
        - keep_all_text: true (default, keeps ALL text)
        - include_diagonal_lines: true (default, includes all orientations)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, process_page_data, page, data.config) for page in data.pages]
        processed_pages = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        total_original_texts = sum(p.stats["original_texts"] for p in processed_pages)
        total_filtered_texts = sum(p.stats["filtered_texts"] for p in processed_pages)
        total_original_lines = sum(p.stats["original_lines"] for p in processed_pages)
        total_filtered_lines = sum(p.stats["filtered_lines"] for p in processed_pages)
        return OutputData(
            pages=processed_pages,
            processing_stats={
                "total_pages": len(data.pages),
                "processing_time_seconds": round(processing_time, 3),
                "total_texts_processed": total_original_texts,
                "total_texts_filtered": total_filtered_texts,
                "total_lines_processed": total_original_lines,
                "total_lines_filtered": total_filtered_lines,
                "filter_config": data.config.dict()
            }
        )
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/pre-filter/batch/")
async def pre_filter_batch(data_batches: List[InputData]):
    """Process multiple data batches concurrently."""
    try:
        tasks = [pre_filter(batch) for batch in data_batches]
        results = await asyncio.gather(*tasks)
        return {"batch_results": results, "total_batches": len(data_batches)}
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True, reload=False)
