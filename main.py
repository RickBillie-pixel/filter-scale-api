from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
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
        
        # Legacy format: direct lines
        all_lines.extend(self.lines)
        
        # New format: lines inside drawings
        if self.drawings and "lines" in self.drawings:
            for line_data in self.drawings["lines"]:
                try:
                    # Convert drawing line format to our Line model
                    line = Line(
                        p1=line_data["p1"],
                        p2=line_data["p2"], 
                        length=line_data["length"],
                        width=line_data.get("width")
                    )
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

class ProcessedPage(BaseModel):
    page_number: int
    texts: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    stats: Dict[str, Any]  # Changed from int to Any to allow nested dicts

class OutputData(BaseModel):
    pages: List[ProcessedPage]
    processing_stats: Dict[str, Any]

# Compiled regex for better performance
NUMERIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?)')

@lru_cache(maxsize=1000)
def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text with caching for repeated texts."""
    match = NUMERIC_PATTERN.search(text)
    return float(match.group(1)) if match else None

def get_midpoint(p1: Dict[str, float], p2: Dict[str, float]) -> Tuple[float, float]:
    """Calculate midpoint between two points."""
    return ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2)

def get_text_midpoint(text: Text) -> Tuple[float, float]:
    """Get text midpoint from position."""
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

def process_page_data(page: Page, config: FilterConfig) -> ProcessedPage:
    """Process a single page - CPU intensive operation."""
    
    # Text Filtering - keep ALL texts or only numeric based on config
    if config.keep_all_text:
        # Keep ALL text types - no filtering
        filtered_texts = page.texts.copy()
        logger.info(f"Keeping all {len(filtered_texts)} texts")
    else:
        # Original behavior - only texts with numeric values
        filtered_texts = [
            t for t in page.texts 
            if extract_numeric_value(t.text) is not None
        ]
        # Sort by numeric value (highest first)
        filtered_texts.sort(
            key=lambda t: extract_numeric_value(t.text) or 0, 
            reverse=True
        )
        logger.info(f"Filtered to {len(filtered_texts)} numeric texts")
    
    # Get all lines from both legacy and new format
    all_lines = page.get_all_lines()
    original_line_count = len(all_lines)
    
    logger.info(f"Processing {original_line_count} lines with min_length={config.min_line_length}")
    
    # Line Filtering - much simpler, no text association
    filtered_lines = []
    
    for i, line in enumerate(all_lines):
        logger.debug(f"Line {i}: length={line.length}")
        
        # Only check line length - no text proximity filtering
        if line.length >= config.min_line_length:
            orientation = assign_orientation(line, config.diagonal_tolerance)
            
            # Include all orientations if diagonal lines are enabled
            if config.include_diagonal_lines or orientation != "diagonal":
                line_dict = line.dict()
                line_dict["orientation"] = orientation
                line_dict["midpoint"] = get_midpoint(line.p1, line.p2)
                filtered_lines.append(line_dict)
                logger.debug(f"Line {i} kept: length={line.length}, orientation={orientation}")
            else:
                logger.debug(f"Line {i} filtered out: diagonal not included")
        else:
            logger.debug(f"Line {i} filtered out: length {line.length} < {config.min_line_length}")
    
    logger.info(f"Filtered to {len(filtered_lines)} lines (min length: {config.min_line_length})")
    

    
    return ProcessedPage(
        page_number=page.page_number,
        texts=[t.dict() for t in filtered_texts],
        lines=unique_lines,
        stats={
            "original_texts": len(page.texts),
            "filtered_texts": len(filtered_texts),
            "original_lines": original_line_count,
            "filtered_lines": len(unique_lines),
            "filter_config_used": {
                "min_line_length": config.min_line_length,
                "keep_all_text": config.keep_all_text,
                "include_diagonal_lines": config.include_diagonal_lines
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
    return {
        "status": "healthy",
        "service": "Pre-Filter API",
        "version": "1.0.0"
    }

@app.post("/debug-input/")
async def debug_input(data: InputData):
    """Debug endpoint to inspect incoming data structure"""
    try:
        debug_info = {
            "total_pages": len(data.pages),
            "config": data.config.dict(),
            "pages_summary": []
        }
        
        for i, page in enumerate(data.pages):
            all_lines = page.get_all_lines()
            
            page_info = {
                "page_number": page.page_number,
                "total_texts": len(page.texts),
                "legacy_lines": len(page.lines),
                "drawings_lines": len(page.drawings.get("lines", [])) if page.drawings else 0,
                "total_lines": len(all_lines),
                "line_lengths": [line.length for line in all_lines[:10]],  # First 10 line lengths
                "lines_above_45": sum(1 for line in all_lines if line.length >= 45),
                "sample_texts": [text.text[:50] for text in page.texts[:5]]  # First 5 texts (truncated)
            }
            debug_info["pages_summary"].append(page_info)
            
            if i >= 2:  # Only show first 3 pages to avoid huge response
                break
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {"error": str(e)}

@app.post("/pre-scale", response_model=OutputData)
@app.post("/pre-filter/", response_model=OutputData)
async def pre_filter(data: InputData):
    """
    Filter vector data and texts for scale processing.
    
    **New filtering behavior:**
    - **Texts**: Keeps ALL text types by default (labels, descriptions, dimensions, etc.)
    - **Lines**: Only lines with length >= 45 (configurable)
    - **No proximity filtering**: Lines don't need to be near text
    
    - **pages**: List of pages containing texts and lines
    - **config**: Optional filtering configuration
        - min_line_length: 45.0 (default, was 10.0)
        - keep_all_text: true (default, keeps ALL text)
        - include_diagonal_lines: true (default, includes all orientations)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Process pages concurrently for better performance
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_page_data, page, data.config)
            for page in data.pages
        ]
        
        processed_pages = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Calculate overall statistics
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
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True,
        reload=False
    )
