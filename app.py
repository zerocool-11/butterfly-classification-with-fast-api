import uvicorn
from fastapi import FastAPI, File, UploadFile
from predic import predic
from read import read
from PIL import Image
import io

app = FastAPI()
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    contents = await file.read()
    images = Image.open(io.BytesIO(contents))
    prediction = predic(images)    
    return prediction

#if __name__ == '__main__':
#    uvicorn.run(app, host='0.0.0.0', port=8000)