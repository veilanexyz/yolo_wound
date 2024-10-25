from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

app = FastAPI()

model_path = './yolo_30_wound.pt'  
model = YOLO(model_path)  

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert("RGB")
    return image

@app.post("/segment")
async def segment_objects(file: UploadFile = File(...)):
    try:
        image = read_imagefile(await file.read()) 

        results = model(image) 

        masks = results[0].masks  

        masks_json = []
        for mask in masks.xy:  
            masks_json.append(mask.tolist())  

        return {"masks": masks_json} 

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
