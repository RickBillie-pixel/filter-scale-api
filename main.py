import os
import json
import logging
import uuid  # Added missing import
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
    description="Pre-filters Vector Drawing API output to include only lines with length >= 50 and texts",
    version="1.0.2"
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

class FilteredText(BaseModel):
    text: str
    position: Dict[str, float]
    bbox: Optional[Dict[str, float]] = None

class FilteredPage(BaseModel):
    page_number: int
    page_size: Dict[str, float]
    texts: List[FilteredText]
    drawings: Dict[str, List]  # Only lines will be included

class FilteredOutput(BaseModel):
    metadata: Dict[str, str]
    pages: List[FilteredPage]
    summary: Dict[str, Any]

@app.post("/pre-filter/")
async def pre_filter(file: UploadFile):
    """Pre-filter the Vector Drawing JSON file to include only lines with length >= 50 and texts"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read and parse the uploaded JSON file
        contents = await file.read()
        input_data = json.loads(contents.decode('utf-8'))
        
        # Save input for debugging
        debug_path = None
        try:
            debug_path = f"/tmp/input_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                json.dump(input_data, f)
            logger.info(f"Saved input for debugging to {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug input: {e}")
        
        # Validate basic structure
        if not input_data.get('pages') or not input_data.get('metadata'):
            raise HTTPException(status_code=400, detail=f"Invalid Vector Drawing JSON structure: {json.dumps({'metadata': input_data.get('metadata'), 'pages': input_data.get('pages')[:1] if input_data.get('pages') else None})}")
        
        if not isinstance(input_data.get('pages'), list):
            raise HTTPException(status_code=400, detail="Pages must be a list")
        
        # Filter data
        filtered_pages = []
        total_lines = 0
        total_texts = 0
        
        for page in input_data['pages']:
            filtered_drawings = {'lines': []}
            
            # Safely access drawings
            drawings = page.get('drawings', {})
            lines = drawings.get('lines', [])
            if not isinstance(lines, list):
                logger.warning(f"Unexpected drawings format for page {page.get('page_number')}: {drawings}")
                continue
            
            # Filter vectors to only lines with length >= 50
            for vec in lines:
                if vec.get('type') == 'line' and vec.get('length', 0) >= 50:
                    try:
                        filtered_vec = FilteredVector(
                            type=vec['type'],
                            p1=vec['p1'],
                            p2=vec['p2'],
                            length=vec.get('length'),
                            orientation=vec.get('orientation')
                        )
                        filtered_drawings['lines'].append(filtered_vec.dict())
                        total_lines += 1
                    except KeyError as e:
                        logger.warning(f"Missing required field in vector {vec}: {e}")
                        continue
            
            # Keep texts, removing non-essential attributes
            filtered_texts = []
            texts = page.get('texts', [])
            if not isinstance(texts, list):
                logger.warning(f"Unexpected texts format for page {page.get('page_number')}: {texts}")
                continue
            
            for txt in texts:
                if 'text' in txt and 'position' in txt:
                    try:
                        filtered_txt = FilteredText(
                            text=txt['text'],
                            position=txt['position'],
                            bbox=txt.get('bbox')
                        )
                        filtered_texts.append(filtered_txt.dict())
                        total_texts += 1
                    except KeyError as e:
                        logger.warning(f"Missing required field in text {txt}: {e}")
                        continue
            
            filtered_page = FilteredPage(
                page_number=page['page_number'],
                page_size=page['page_size'],
                texts=filtered_texts,
                drawings=filtered_drawings
            )
            filtered_pages.append(filtered_page.dict())
        
        # Prepare filtered summary
        filtered_summary = {
            'total_pages': len(filtered_pages),
            'total_texts': total_texts,
            'total_lines': total_lines,
            'total_rectangles': 0,
            'total_curves': 0,
            'total_polygons': 0,
            'dimensions_found': input_data['summary'].get('dimensions_found', 0) if 'summary' in input_data else 0,
            'file_size_mb': input_data['summary'].get('file_size_mb', 0) if 'summary' in input_data else 0,
            'processing_time_ms': input_data['summary'].get('processing_time_ms', 0) if 'summary' in input_data else 0
        }
        
        # Build response
        filtered_output = FilteredOutput(
            metadata=input_data['metadata'],
            pages=filtered_pages,
            summary=filtered_summary
        )
        
        logger.info(f"Filtered output: {total_lines} lines, {total_texts} texts")
        
        return filtered_output.dict()
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during pre-filtering: {e}", exc_info=True)
        # Save full input for debugging on error
        if debug_path is None:
            try:
                debug_path = f"/tmp/error_input_{uuid.uuid4()}.json"
                with open(debug_path, 'w') as f:
                    json.dump(input_data, f)
                logger.info(f"Saved error input to {debug_path}")
            except Exception as save_error:
                logger.warning(f"Failed to save error input: {save_error}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.2"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
