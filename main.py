import os
import json
import logging
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
    description="Pre-filters Vector Drawing API output to include only lines and texts, removing non-essential attributes",
    version="1.0.0"
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
    """Pre-filter the Vector Drawing JSON file to include only lines and texts"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read and parse the uploaded JSON file
        contents = await file.read()
        input_data = json.loads(contents.decode('utf-8'))
        
        # Validate basic structure
        if not input_data.get('pages') or not input_data.get('metadata'):
            raise HTTPException(status_code=400, detail="Invalid Vector Drawing JSON structure")
        
        # Filter data
        filtered_pages = []
        total_lines = 0
        total_texts = 0
        
        for page in input_data['pages']:
            filtered_drawings = {'lines': []}
            
            # Filter vectors to only lines
            for vec in page.get('drawings', {}).get('lines', []):
                if vec.get('type') == 'line':
                    filtered_vec = FilteredVector(
                        type=vec['type'],
                        p1=vec['p1'],
                        p2=vec['p2'],
                        length=vec.get('length'),
                        orientation=vec.get('orientation')
                    )
                    filtered_drawings['lines'].append(filtered_vec.dict())
                    total_lines += 1
            
            # Keep texts, removing non-essential attributes like source, dimension_info
            filtered_texts = []
            for txt in page.get('texts', []):
                filtered_txt = FilteredText(
                    text=txt['text'],
                    position=txt['position'],
                    bbox=txt.get('bbox')
                )
                filtered_texts.append(filtered_txt.dict())
                total_texts += 1
            
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
            'total_rectangles': 0,  # Excluded
            'total_curves': 0,  # Excluded
            'total_polygons': 0,  # Excluded
            'dimensions_found': input_data['summary']['dimensions_found'],  # Keep original
            'file_size_mb': input_data['summary']['file_size_mb'],
            'processing_time_ms': input_data['summary']['processing_time_ms']
        }
        
        # Build response
        filtered_output = FilteredOutput(
            metadata=input_data['metadata'],
            pages=filtered_pages,
            summary=filtered_summary
        )
        
        logger.info(f"Filtered output: {total_lines} lines, {total_texts} texts")
        
        return filtered_output.dict()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during pre-filtering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
