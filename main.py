from fastapi import FastAPI,HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import boto3
import uuid
import aiohttp
import asyncio
from pathlib import Path
import uvicorn
import cv2
from typing import List
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError


s3_client = boto3.client(
    's3',
    endpoint_url='',
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name=''
)
BUCKET_NAME = ""

app = FastAPI()

model = YOLO('best.pt')

UPLOAD_DIR = Path("predictions")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def calculate_defect_severity(defect_type: str, confidence: float, bbox: list) -> str:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    
    SMALL_AREA = 0.01  
    MEDIUM_AREA = 0.05
    LARGE_AREA = 0.1   
    
    base_severity = {
        "яма": 3,        
        "выбоина": 2,    
        "трещина": 1,   
        "лужа": 1     
    }

    severity_score = base_severity.get(defect_type, 1)
    
    if area > LARGE_AREA:
        severity_score += 2
    elif area > MEDIUM_AREA:
        severity_score += 1
    elif area < SMALL_AREA:
        severity_score -= 1
        

    if confidence < 0.3:
        severity_score -= 1
    elif confidence > 0.8:
        severity_score += 1

    if severity_score <= 1:
        return "low"
    elif severity_score == 2:
        return "middle"
    elif severity_score == 3:
        return "high"
    else:
        return "critical"

def get_overall_severity(predictions: list) -> str:
    if not predictions:
        return "low"

    severity_counts = {
        "critical": 0,
        "high": 0,
        "middle": 0,
        "low": 0
    }
    
    for pred in predictions:
        severity = calculate_defect_severity(
            pred['class'],
            pred['confidence'],
            pred['bbox']
        )
        severity_counts[severity] += 1
    

    if severity_counts["critical"] > 0:
        return "critical"
    elif severity_counts["high"] > 1:
        return "critical"
    elif severity_counts["high"] == 1:
        return "high"
    elif severity_counts["middle"] > 2:
        return "high"
    elif severity_counts["middle"] > 0:
        return "middle"
    else:
        return "low"
def upload_to_s3(image, file_name):

    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)  

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=f"images/{file_name}",
            Body=img_byte_arr,
            ContentType='image/jpeg'
        )
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to S3")
class ImageBatchRequest(BaseModel):
    files: List[str]

class PredictionResponse(BaseModel):
    url: str
    id: str
    predictions: List[dict]
    type: str
    image_url: str
    success: bool
    error: str = None

async def download_image(session: aiohttp.ClientSession, url: str) -> tuple[str, Image.Image | None, str | None]:
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return url, None, f"Failed to download image: HTTP {response.status}"
            image_data = await response.read()
            return url, Image.open(io.BytesIO(image_data)), None
    except Exception as e:
        return url, None, str(e) 
async def process_single_image(url: str, image: Image.Image) -> PredictionResponse:
    try:
        file_uuid = str(uuid.uuid4())
        
        results = model(image)[0]
        

        predictions = []
        for box in results.boxes:
            box_data = box.data[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box_data[0:4])
            
            prediction = {
                'class': results.names[class_id],
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            }
            prediction['severity'] = calculate_defect_severity(
                prediction['class'],
                confidence,
                [x1, y1, x2, y2]
            )
            predictions.append(prediction)
        
        overall_severity = get_overall_severity(predictions)
        
        plotted_image = Image.fromarray(results.plot())
        upload_to_s3(plotted_image, f"{file_uuid}.jpg")
        
        return PredictionResponse(
            url=url,
            id=file_uuid,
            predictions=predictions,
            type=overall_severity,
            image_url=f"{file_uuid}.jpg",
            success=True
        )
    except Exception as e:
        return PredictionResponse(
            url=url,
            id="",
            predictions=[],
            type="low",
            image_url="",
            success=False,
            error=str(e)
        )
@app.post("/predict", response_model=List[PredictionResponse])
async def predict_batch(request: ImageBatchRequest):
    async with aiohttp.ClientSession() as session:
        download_tasks = [download_image(session, url) for url in request.files]
        downloaded_images = await asyncio.gather(*download_tasks)
        
        processing_tasks = []
        for url, image, error in downloaded_images:
            if image is not None:
                processing_tasks.append(process_single_image(url, image))
            else:
                processing_tasks.append(asyncio.sleep(0, result=PredictionResponse(
                    url=url,
                    id="",
                    predictions=[],
                    type="low",
                    image_url="",
                    success=False,
                    error=error
                )))
        
        results = await asyncio.gather(*processing_tasks)
        return results

class RTSPRequest(BaseModel):
    rtsps: List[str]

def capture_rtsp_frame(rtsp_url: str) -> Image.Image:
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise Exception(f"Cannot open RTSP stream: {rtsp_url}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception(f"Cannot read frame from stream: {rtsp_url}")
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error capturing frame from {rtsp_url}: {str(e)}")

async def process_single_stream(rtsp_url: str, model: YOLO) -> dict:
    image = await asyncio.to_thread(capture_rtsp_frame, rtsp_url)
    file_uuid = str(uuid.uuid4())

    results = model(image)[0]
    
    predictions = []
    for box in results.boxes:
        box_data = box.data[0]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box_data[0:4])
        
        prediction = {
            'class': results.names[class_id],
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2]
        }
        prediction['severity'] = calculate_defect_severity(
            prediction['class'],
            confidence,
            [x1, y1, x2, y2]
        )
        predictions.append(prediction)
    
    overall_severity = get_overall_severity(predictions)

    plotted_image = Image.fromarray(results.plot())
    upload_to_s3(plotted_image, file_uuid+".jpg")
    
    return {
        'rtsp_url': rtsp_url,
        'id': file_uuid,
        'predictions': predictions,
        'type': overall_severity,
        'image_url': f'{file_uuid}.jpg'
    }

@app.post("/rtsp")
async def process_rtsp_streams(request: RTSPRequest):
    try:
        tasks = [process_single_stream(rtsp_url, model) for rtsp_url in request.rtsps]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'rtsp_url': request.rtsps[i],
                    'error': str(result),
                    'success': False
                })
            else:
                result['success'] = True
                processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RTSP streams: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)