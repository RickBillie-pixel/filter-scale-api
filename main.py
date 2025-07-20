from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import re
import math
from typing import List, Dict, Optional, Any, Union
import time
import logging

app = FastAPI(title="Pre-Scale Filter API", description="Detects scales from vector drawing JSON", version="1.0.1")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class TextItem(BaseModel):
    text: str
    position: Optional[Dict[str, float]] = None
    bbox: Dict[str, float]

class LineItem(BaseModel):
    type: str
    points: Optional[List[Dict[str, float]]] = None  # For beziers
    p1: Optional[Dict[str, float]] = None  # For straight lines
    p2: Optional[Dict[str, float]] = None
    length: Optional[float] = None

class PageItem(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    texts: List[TextItem]
    drawings: Dict[str, List[Dict[str, Any]]]  # More flexible structure

class InputData(BaseModel):
    metadata: Optional[Dict] = None
    pages: List[PageItem]

class ScaleMatch(BaseModel):
    text_value: float
    line_length_points: float
    calculated_scale: float
    distance_to_text_points: float
    ratio: float = Field(default=1.0)

class OrientationScale(BaseModel):
    points_per_meter: float
    confidence: float
    validation_count: int
    lines_used: List[ScaleMatch]

class OutputData(BaseModel):
    status: str
    scales: Dict[str, OrientationScale]
    dimension_lines: List[Dict] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_stats: Dict

# Regex patterns for dimensions - more comprehensive
DIMENSION_PATTERNS = [
    r'^\d{2,5}$',  # Pure numbers 10-99999
    r'^\d+\.\d+$',  # Decimals like 123.45
    r'^\d+,\d+$',  # Comma decimals like 123,45
    r'^\d+\.?\d*\s*(?:mm|cm|m|MM|CM|M)$',  # With units
    r'^\d+[xX×]\d+$',  # Dimensions like 67X114
    r'^\d+[\s]*[xX×][\s]*\d+$',  # Dimensions with spaces
]

def extract_numeric_value(text: str) -> Optional[float]:
    """Extract and convert dimension to mm."""
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip().replace(',', '.').lower()
    
    # Skip very short text or non-numeric text
    if len(text) < 2:
        return None
    
    for pattern in DIMENSION_PATTERNS:
        if re.match(pattern, text):
            try:
                if 'x' in text or '×' in text:
                    parts = re.split(r'[x×]', text)
                    if len(parts) >= 2:
                        return float(re.sub(r'[^\d.]', '', parts[0]))
                
                # Extract number and check for units
                number_str = re.sub(r'[^\d.]', '', text)
                if not number_str:
                    continue
                    
                value = float(number_str)
                
                if 'mm' in text:
                    return value
                elif 'cm' in text:
                    return value * 10
                elif 'm' in text and 'mm' not in text:
                    return value * 1000
                else:
                    # Default assumption: mm for reasonable building dimensions
                    if 10 <= value <= 50000:  # Reasonable range in mm
                        return value
                    
            except (ValueError, IndexError):
                continue
                
    return None

def calculate_length(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Calculate Euclidean distance between two points."""
    try:
        return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
    except (KeyError, TypeError):
        return 0.0

def is_straight_bezier(points: List[Dict[str, float]], tolerance: float = 0.02) -> bool:
    """Check if bezier curve is essentially a straight line."""
    if len(points) != 4:
        return False
    
    try:
        p0, p1, p2, p3 = points
        total_length = calculate_length(p0, p3)
        
        if total_length < 1:  # Too short to be meaningful
            return False
        
        # Check if control points are reasonably aligned with start/end
        control_deviation = (calculate_length(p0, p1) + calculate_length(p1, p2) + 
                           calculate_length(p2, p3)) / total_length
        
        return control_deviation < (1 + tolerance)
    except (KeyError, TypeError):
        return False

def filter_lines(drawings: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract and filter the longest meaningful lines."""
    all_lines = []
    
    # Process different line types from drawings
    for line_type, items in drawings.items():
        if not isinstance(items, list):
            continue
            
        for item in items:
            if not isinstance(item, dict):
                continue
                
            try:
                line_data = None
                
                if item.get('type') == 'line' and 'p1' in item and 'p2' in item:
                    p1, p2 = item['p1'], item['p2']
                    length = item.get('length') or calculate_length(p1, p2)
                    line_data = {'p1': p1, 'p2': p2, 'length': length}
                    
                elif item.get('type') == 'bezier' and 'points' in item:
                    points = item['points']
                    if len(points) >= 2 and is_straight_bezier(points):
                        p1, p2 = points[0], points[-1]
                        length = calculate_length(p1, p2)
                        line_data = {'p1': p1, 'p2': p2, 'length': length}
                
                if line_data and line_data['length'] > 10:  # Minimum meaningful length
                    p1, p2 = line_data['p1'], line_data['p2']
                    dx = abs(p2['x'] - p1['x'])
                    dy = abs(p2['y'] - p1['y'])
                    
                    # Classify orientation with more tolerance
                    if dx > dy * 5:  # Horizontal
                        line_data['orientation'] = 'horizontal'
                        all_lines.append(line_data)
                    elif dy > dx * 5:  # Vertical
                        line_data['orientation'] = 'vertical'
                        all_lines.append(line_data)
                        
            except (KeyError, TypeError, ZeroDivisionError) as e:
                logger.debug(f"Skipping invalid line item: {e}")
                continue
    
    # Sort by length and return top 100 for analysis
    all_lines.sort(key=lambda l: l['length'], reverse=True)
    logger.info(f"Found {len(all_lines)} valid lines for analysis")
    return all_lines[:100]

def get_text_position(text: TextItem) -> tuple:
    """Get center position of text item."""
    try:
        bbox = text.bbox
        return ((bbox['x0'] + bbox['x1']) / 2, (bbox['y0'] + bbox['y1']) / 2)
    except (KeyError, TypeError):
        return (0, 0)

def get_line_midpoint(line: Dict) -> tuple:
    """Get midpoint of a line."""
    try:
        p1, p2 = line['p1'], line['p2']
        return ((p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2)
    except (KeyError, TypeError):
        return (0, 0)

def distance_between_points(pos1: tuple, pos2: tuple) -> float:
    """Calculate distance between two positions."""
    try:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    except (TypeError, ValueError):
        return float('inf')

def find_scale_matches(texts: List[TextItem], lines: List[Dict], orientation: str, max_distance: float) -> List[ScaleMatch]:
    """Find matches between dimension texts and lines of specific orientation."""
    matches = []
    
    orientation_lines = [l for l in lines if l['orientation'] == orientation]
    logger.info(f"Processing {len(orientation_lines)} {orientation} lines with {len(texts)} texts")
    
    for text in texts:
        text_value = extract_numeric_value(text.text)
        if text_value is None or text_value < 10 or text_value > 50000:  # Reasonable range
            continue
            
        text_pos = get_text_position(text)
        
        for line in orientation_lines:
            line_mid = get_line_midpoint(line)
            distance = distance_between_points(text_pos, line_mid)
            
            if distance <= max_distance:
                # Convert text_value from mm to meters for scale calculation
                text_meters = text_value / 1000.0
                if text_meters > 0:
                    scale = line['length'] / text_meters  # points per meter
                    
                    matches.append(ScaleMatch(
                        text_value=text_value,
                        line_length_points=line['length'],
                        calculated_scale=scale,
                        distance_to_text_points=distance,
                        ratio=1.0
                    ))
    
    # Sort by distance to text (closest first)
    matches.sort(key=lambda m: m.distance_to_text_points)
    logger.info(f"Found {len(matches)} potential {orientation} matches")
    return matches[:20]  # Keep top 20 matches

def validate_scale_consistency(matches: List[ScaleMatch], min_matches: int = 3) -> Optional[OrientationScale]:
    """Validate scale consistency and return best scale."""
    if len(matches) < min_matches:
        logger.info(f"Insufficient matches: {len(matches)} < {min_matches}")
        return None
    
    # Sort by scale value to find clusters
    matches.sort(key=lambda m: m.calculated_scale)
    
    # Find the most consistent cluster
    best_scale = None
    best_confidence = 0
    
    for i in range(len(matches) - min_matches + 1):
        cluster = matches[i:i + min_matches]
        scales = [m.calculated_scale for m in cluster]
        
        avg_scale = sum(scales) / len(scales)
        if avg_scale <= 0:
            continue
            
        # Calculate deviation
        max_deviation = max(abs(s - avg_scale) for s in scales) / avg_scale * 100
        
        if max_deviation <= 15:  # Allow up to 15% deviation
            confidence = max(60, 100 - (max_deviation * 2))  # Scale confidence
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_scale = OrientationScale(
                    points_per_meter=round(avg_scale, 2),
                    confidence=round(confidence, 1),
                    validation_count=len(cluster),
                    lines_used=cluster
                )
    
    return best_scale

async def process_scale_data(input_data: InputData) -> OutputData:
    """Main processing function."""
    start_time = time.time()
    
    try:
        # Collect all texts and drawings from all pages
        all_texts = []
        all_drawings = {}
        
        for page in input_data.pages:
            all_texts.extend(page.texts)
            for key, items in page.drawings.items():
                if key not in all_drawings:
                    all_drawings[key] = []
                all_drawings[key].extend(items)
        
        logger.info(f"Processing {len(all_texts)} texts and {sum(len(items) for items in all_drawings.values())} drawing items")
        
        # Filter and extract meaningful lines
        filtered_lines = filter_lines(all_drawings)
        
        if not filtered_lines:
            raise HTTPException(status_code=400, detail="No valid lines found for scale detection")
        
        # Calculate dynamic max distance based on drawing size
        avg_length = sum(l['length'] for l in filtered_lines) / len(filtered_lines)
        max_distance = max(30, min(150, avg_length * 0.15))
        
        logger.info(f"Using max distance: {max_distance} for text-line matching")
        
        # Find matches for both orientations
        horizontal_matches = find_scale_matches(all_texts, filtered_lines, 'horizontal', max_distance)
        vertical_matches = find_scale_matches(all_texts, filtered_lines, 'vertical', max_distance)
        
        # Validate and determine scales
        horizontal_scale = validate_scale_consistency(horizontal_matches)
        vertical_scale = validate_scale_consistency(vertical_matches)
        
        # Prepare results
        scales = {}
        warnings = []
        
        if horizontal_scale:
            scales['horizontal'] = horizontal_scale
        else:
            warnings.append("Could not determine reliable horizontal scale")
            
        if vertical_scale:
            scales['vertical'] = vertical_scale
        else:
            warnings.append("Could not determine reliable vertical scale")
        
        # Check for significant deviation between scales
        if horizontal_scale and vertical_scale:
            h_scale = horizontal_scale.points_per_meter
            v_scale = vertical_scale.points_per_meter
            deviation = abs(h_scale - v_scale) / max(h_scale, v_scale) * 100
            
            if deviation > 5:
                warnings.append(f"Significant scale deviation between orientations: {deviation:.1f}%")
        
        # If no scales found, provide fallback
        if not scales:
            # Use a reasonable default scale based on typical architectural drawings
            default_scale = 50.0  # 50 points per meter (roughly 1:72 scale)
            scales['horizontal'] = OrientationScale(
                points_per_meter=default_scale,
                confidence=30.0,
                validation_count=0,
                lines_used=[]
            )
            scales['vertical'] = scales['horizontal']
            warnings.append("Using default scale - no reliable matches found")
        
        processing_time = (time.time() - start_time) * 1000
        
        stats = {
            "lines_processed": len(filtered_lines),
            "texts_processed": len(all_texts),
            "dimension_texts_found": len([t for t in all_texts if extract_numeric_value(t.text) is not None]),
            "horizontal_matches": len(horizontal_matches),
            "vertical_matches": len(vertical_matches),
            "processing_time_ms": round(processing_time, 2)
        }
        
        return OutputData(
            status="success",
            scales=scales,
            warnings=warnings,
            processing_stats=stats
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/pre-scale", response_model=OutputData)
async def pre_scale_endpoint(request: Request, data: InputData):
    """Main endpoint for scale processing - accepts JSON data."""
    logger.info(f"Processing scale request from {request.client.host if request.client else 'unknown'}")
    return await process_scale_data(data)

@app.post("/pre-scale/", response_model=OutputData)
async def pre_scale_endpoint_slash(request: Request, data: InputData):
    """Alternative endpoint with trailing slash."""
    return await pre_scale_endpoint(request, data)

@app.post("/upload-pre-scale")
async def upload_pre_scale(file: UploadFile = File(...)):
    """Alternative endpoint for file upload."""
    try:
        content = await file.read()
        data = json.loads(content)
        input_data = InputData(**data)
        return await process_scale_data(input_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
        "version": "1.0.1"
    }

@app.get("/")
async def root():
    return {
        "title": "Pre-Scale Filter API",
        "description": "Detects scales from vector drawing JSON data",
        "version": "1.0.1",
        "endpoints": {
            "/pre-scale": "POST - Process scale data (JSON)",
            "/upload-pre-scale": "POST - Process scale data (file upload)",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
