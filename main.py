from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import re
import math
from typing import List, Dict, Optional
import time
import logging
from ratelimit import limits, sleep_and_retry
from functools import lru_cache

app = FastAPI(title="Pre-Scale Filter API", description="Detects scales from vector drawing JSON", version="1.0.0")

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
    drawings: Dict[str, List[LineItem]]

class InputData(BaseModel):
    metadata: Optional[Dict] = None
    pages: List[PageItem]

class ScaleMatch(BaseModel):
    text_value: float
    line_length_points: float
    calculated_scale: float
    distance_to_text_points: float
    ratio: float

class OrientationScale(BaseModel):
    points_per_meter: float
    confidence: float
    validation_count: int
    lines_used: List[ScaleMatch]

class OutputData(BaseModel):
    status: str
    scales: Dict[str, OrientationScale]
    dimension_lines: List[Dict]  # Placeholder for future
    warnings: List[str]
    processing_stats: Dict

# Regex patterns for dimensions
DIMENSION_PATTERNS = [
    r'^\d{2,5}$',  # Pure numbers 10-99999
    r'^\d+\.\d+$',  # Decimals
    r'^\d+(?:\s+\d+)*$',  # Multi-numbers
    r'^\d+\.?\d*\s*(?:mm|cm|m)$',  # With units
    r'^\d+[xX×]\d+$'  # e.g., 67X114
]

def extract_numeric_value(text: str) -> Optional[float]:
    """Extract and convert dimension to mm."""
    text = text.strip().lower()
    for pattern in DIMENSION_PATTERNS:
        if re.match(pattern, text):
            if 'x' in text or '×' in text:
                parts = re.split(r'[x×]', text)
                return float(parts[0])  # Use first part for simplicity, assume width/height
            if 'mm' in text:
                return float(re.sub(r'[^\d.]', '', text))
            elif 'cm' in text:
                return float(re.sub(r'[^\d.]', '', text)) * 10
            elif 'm' in text:
                return float(re.sub(r'[^\d.]', '', text)) * 1000
            else:
                return float(text)  # Assume mm if no unit
    return None

def calculate_length(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Calculate Euclidean length."""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_straight_bezier(points: List[Dict[str, float]]) -> bool:
    """Check if bezier is essentially straight (collinear points within 1%)."""
    if len(points) != 4:
        return False
    p0, p1, p2, p3 = points
    # Check if control points align
    dx1 = p1['x'] - p0['x']
    dy1 = p1['y'] - p0['y']
    dx2 = p3['x'] - p2['x']
    dy2 = p3['y'] - p2['y']
    return abs(dx1 * dy2 - dx2 * dy1) < 0.01 * calculate_length(p0, p3)

def filter_lines(drawings: Dict[str, List[LineItem]]) -> List[Dict]:
    """Filter top 50 longest straight lines, classify orientation."""
    all_lines = []
    for item in drawings.get('lines', []):
        if item.type == 'line':
            length = item.length or calculate_length(item.p1, item.p2)
            if length > 10:
                p1, p2 = item.p1, item.p2
                dx = abs(p2['x'] - p1['x'])
                dy = abs(p2['y'] - p1['y'])
                orientation = 'horizontal' if dx > dy * 10 else 'vertical' if dy > dx * 10 else 'diagonal'
                if orientation != 'diagonal':
                    all_lines.append({'p1': p1, 'p2': p2, 'length': length, 'orientation': orientation})
        elif item.type == 'bezier' and is_straight_bezier(item.points):
            p1, p3 = item.points[0], item.points[3]
            length = calculate_length(p1, p3)
            if length > 10:
                dx = abs(p3['x'] - p1['x'])
                dy = abs(p3['y'] - p1['y'])
                orientation = 'horizontal' if dx > dy * 10 else 'vertical' if dy > dx * 10 else 'diagonal'
                if orientation != 'diagonal':
                    all_lines.append({'p1': p1, 'p2': p3, 'length': length, 'orientation': orientation})
    all_lines.sort(key=lambda l: l['length'], reverse=True)
    return all_lines[:50]

def midpoint(line: Dict) -> tuple:
    """Get midpoint of line."""
    return ((line['p1']['x'] + line['p2']['x']) / 2, (line['p1']['y'] + line['p2']['y']) / 2)

def distance_to_midpoint(text_pos: tuple, line_mid: tuple) -> float:
    """Euclidean distance to midpoint."""
    return math.sqrt((text_pos[0] - line_mid[0])**2 + (text_pos[1] - line_mid[1])**2)

@lru_cache(maxsize=128)
def match_text_to_lines(text: TextItem, lines: List[Dict], orientation: str, max_distance: float) -> List[Dict]:
    """Match text to closest lines of given orientation."""
    text_value = extract_numeric_value(text.text)
    if text_value is None:
        return []
    text_pos = ((text.bbox['x0'] + text.bbox['x1']) / 2, (text.bbox['y0'] + text.bbox['y1']) / 2)
    matches = []
    for line in [l for l in lines if l['orientation'] == orientation]:
        dist = distance_to_midpoint(text_pos, midpoint(line))
        if dist <= max_distance:
            scale = line['length'] / (text_value / 1000) if text_value > 0 else 0
            ratio = scale / 50 if scale > 0 else 0  # Placeholder nominal; adjust based on expected
            matches.append({
                'text_value': text_value,
                'line_length_points': line['length'],
                'calculated_scale': scale,
                'distance_to_text_points': dist,
                'ratio': 1.0  # Simplified; in full, compute relative to avg
            })
    matches.sort(key=lambda m: m['distance_to_text_points'])
    return matches[:3]

def determine_scale(matches: List[Dict], orientation: str) -> Optional[OrientationScale]:
    """Validate and average top 3 matches by value."""
    if len(matches) < 3:
        return None  # Require min 3 for validation
    matches.sort(key=lambda m: m['text_value'], reverse=True)  # Top by dimension value
    top3 = matches[:3]
    scales = [m['calculated_scale'] for m in top3]
    avg_scale = sum(scales) / len(scales)
    deviation = max(abs(s - avg_scale) for s in scales) / avg_scale * 100
    if deviation > 2:
        return None  # Fail validation
    # Confidence calculation
    conf = 100 - (deviation * 5)  # Base 100, deduct for deviation
    conf = min(100, max(90, conf))  # Clamp
    return OrientationScale(
        points_per_meter=round(avg_scale, 2),
        confidence=round(conf, 1),
        validation_count=3,
        lines_used=top3
    )

@sleep_and_retry
@limits(calls=10, period=60)  # 10 req/min
async def process_data(input_data: InputData) -> OutputData:
    start_time = time.time()
    all_texts = [text for page in input_data.pages for text in page.texts]
    all_drawings = {k: [item for page in input_data.pages for item in page.drawings.get(k, [])] for k in ['lines']}
    filtered_lines = filter_lines(all_drawings)
    avg_length = sum(l['length'] for l in filtered_lines) / len(filtered_lines) if filtered_lines else 300
    max_distance = max(50, min(100, avg_length * 0.1))
    
    horiz_matches = []
    vert_matches = []
    for text in all_texts:
        horiz_matches.extend(match_text_to_lines(text, tuple(filtered_lines), 'horizontal', max_distance))  # tuple for cache
        vert_matches.extend(match_text_to_lines(text, tuple(filtered_lines), 'vertical', max_distance))
    
    horiz_scale = determine_scale(horiz_matches, 'horizontal')
    vert_scale = determine_scale(vert_matches, 'vertical')
    
    if not horiz_scale or not vert_scale:
        raise HTTPException(status_code=400, detail="Insufficient valid matches for scale determination")
    
    deviation = abs(horiz_scale.points_per_meter - vert_scale.points_per_meter) / max(horiz_scale.points_per_meter, vert_scale.points_per_meter) * 100
    warnings = ["Significant scale deviation between orientations (>2%)"] if deviation > 2 else []
    
    stats = {
        "lines_processed": len(filtered_lines),
        "texts_processed": len(all_texts),
        "dimension_lines_found": len([t for t in all_texts if extract_numeric_value(t.text) is not None]),
        "processing_time_ms": round((time.time() - start_time) * 1000)
    }
    
    return OutputData(
        status="success",
        scales={"horizontal": horiz_scale, "vertical": vert_scale},
        dimension_lines=[],  # Future expansion
        warnings=warnings,
        processing_stats=stats
    )

@app.post("/pre-scale", response_model=OutputData)
async def pre_scale(request: Request, data: InputData):
    logger.info(f"Processing request from {request.client.host}")
    return await process_data(data)

@app.get("/scale/{orientation}")
async def get_scale(orientation: str):
    # Placeholder; in production, cache or store scales
    if orientation not in ["horizontal", "vertical"]:
        raise HTTPException(status_code=400, detail="Invalid orientation")
    # Simulate retrieval
    return {"message": f"Scale for {orientation} retrieved (implement storage for real use)"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
