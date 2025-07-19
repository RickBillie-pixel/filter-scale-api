import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Pre-Filter API",
    description="Pre-filters Vector Drawing API output to include only lines with length >= 65 and texts",
    version="1.0.4"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilteredVector(BaseModel):
    type: str
    p1: Dict[str, float]
    p2: Dict[str, float]
    length: Optional[float] = None
    orientation: Optional[str] = None
    width: Optional[float] = None
    opacity: Optional[float] = None

class FilteredText(BaseModel):
    text: str
    position: Dict[str, float]
    bbox: Optional[Dict[str, float]] = None
    page_number: Optional[int] = None
    source: Optional[str] = None

class FilteredPage(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    texts: List[FilteredText]
    drawings: Dict[str, List]  # Only lines will be included

class FilteredOutput(BaseModel):
    metadata: Dict[str, Any]
    pages: List[FilteredPage]
    summary: Dict[str, Any]

def parse_input_data(contents: str) -> Dict:
    """Parse the input JSON string, handling potential nested JSON."""
    try:
        data = json.loads(contents)
        if isinstance(data, str):
            logger.warning("Input is a string, attempting to parse again")
            data = json.loads(data)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {str(e)}")

@app.post("/pre-filter/")
async def pre_filter(file: UploadFile):
    """Pre-filter the Vector Drawing JSON file to include only lines with length >= 65 and texts"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read and parse the uploaded JSON file
        contents = (await file.read()).decode('utf-8')
        input_data = parse_input_data(contents)
        
        # Save input for debugging
        debug_path = None
        try:
            debug_path = f"/tmp/input_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                json.dump(input_data, f, indent=2)
            logger.info(f"Saved input for debugging to {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug input: {e}")
        
        # Check if this is the old format from Master API
        if 'vector_data' in input_data and 'texts' in input_data:
            logger.error("Received old format from Master API, this should be Vector Drawing API format")
            raise HTTPException(
                status_code=400, 
                detail="Invalid format: Expected Vector Drawing API output with 'metadata', 'pages', and 'summary'"
            )
        
        # Validate Vector Drawing API structure
        if not input_data.get('pages') or not input_data.get('metadata'):
            logger.error(f"Invalid structure: pages={type(input_data.get('pages'))}, metadata={type(input_data.get('metadata'))}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid Vector Drawing JSON structure: missing pages or metadata"
            )
        
        if not isinstance(input_data.get('pages'), list):
            logger.error(f"Pages is not a list: {type(input_data.get('pages'))}")
            raise HTTPException(status_code=400, detail="Pages must be a list")
        
        # Filter data
        filtered_pages = []
        total_lines = 0
        total_texts = 0
        filtered_lines = 0
        
        for page in input_data['pages']:
            # Initialize filtered drawings with only lines
            filtered_drawings = {'lines': []}
            
            # Safely access drawings
            drawings = page.get('drawings', {})
            if not isinstance(drawings, dict):
                logger.warning(f"Drawings is not a dict for page {page.get('page_number')}: {type(drawings)}")
                drawings = {}
            
            lines = drawings.get('lines', [])
            if not isinstance(lines, list):
                logger.warning(f"Lines is not a list for page {page.get('page_number')}: {type(lines)}")
                lines = []
            
            # Filter lines with length >= 65
            for line in lines:
                try:
                    # Check if it's a line and has required fields
                    if (line.get('type') == 'line' and 
                        'p1' in line and 'p2' in line and 
                        line.get('length', 0) >= 65):
                        
                        filtered_vec = {
                            'type': line['type'],
                            'p1': line['p1'],
                            'p2': line['p2'],
                            'length': line.get('length')
                        }
                        
                        # Add optional fields if present
                        if 'orientation' in line:
                            filtered_vec['orientation'] = line['orientation']
                        if 'width' in line:
                            filtered_vec['width'] = line['width']
                        if 'opacity' in line:
                            filtered_vec['opacity'] = line['opacity']
                        
                        filtered_drawings['lines'].append(filtered_vec)
                        filtered_lines += 1
                    total_lines += 1
                except Exception as e:
                    logger.warning(f"Error processing line {line}: {e}")
                    continue
            
            # Process texts
            filtered_texts = []
            texts = page.get('texts', [])
            if not isinstance(texts, list):
                logger.warning(f"Texts is not a list for page {page.get('page_number')}: {type(texts)}")
                texts = []
            
            for txt in texts:
                try:
                    if 'text' in txt and 'position' in txt:
                        filtered_txt = {
                            'text': txt['text'],
                            'position': txt['position']
                        }
                        
                        # Add optional fields if present
                        if 'bbox' in txt:
                            filtered_txt['bbox'] = txt['bbox']
                        if 'page_number' in txt:
                            filtered_txt['page_number'] = txt['page_number']
                        if 'source' in txt:
                            filtered_txt['source'] = txt['source']
                        
                        filtered_texts.append(filtered_txt)
                        total_texts += 1
                except Exception as e:
                    logger.warning(f"Error processing text {txt}: {e}")
                    continue
            
            # Build filtered page
            filtered_page = {
                'page_number': page.get('page_number', 1),
                'page_size': page.get('page_size', {'width': 0, 'height': 0}),
                'texts': filtered_texts,
                'drawings': filtered_drawings
            }
            
            filtered_pages.append(filtered_page)
        
        # Prepare filtered summary
        original_summary = input_data.get('summary', {})
        filtered_summary = {
            'total_pages': len(filtered_pages),
            'total_texts': total_texts,
            'total_lines': filtered_lines,  # Only filtered lines
            'original_lines': total_lines,   # Total lines before filtering
            'total_rectangles': 0,
            'total_curves': 0,
            'total_polygons': 0,
            'dimensions_found': original_summary.get('dimensions_found', 0),
            'file_size_mb': original_summary.get('file_size_mb', 0),
            'processing_time_ms': original_summary.get('processing_time_ms', 0)
        }
        
        # Build response with same structure as input
        filtered_output = {
            'metadata': input_data.get('metadata', {}),
            'pages': filtered_pages,
            'summary': filtered_summary
        }
        
        logger.info(f"Filtering complete: {filtered_lines} lines kept out of {total_lines} (min length: 65), {total_texts} texts")
        
        return filtered_output
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during pre-filtering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.4", "min_line_length": 65}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
