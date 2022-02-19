from frame_ext import frameExt
from face_ext import faceExt
from detect import detect
import wget
from fastapi import FastAPI
from pydantic import BaseModel
import os


app = FastAPI()

class request_body(BaseModel):
    vid : str = 'https://drive.google.com/u/0/uc?id=1shkvsFW1pcQN2GPPSr605w-ZLWJIIf-Z&export=download'
    nframes: int = 100
    type : str = 'frame'

@app.post('/predict')
def predict(args : request_body):
    print('------downloading------')
    if os.path.isfile('video/tmp.mp4'):
        os.remove('video/tmp.mp4')
    wget.download(args.vid, 'video/tmp.mp4')
    frameExt('video/tmp.mp4', args.nframes)
    faceExt(f'frames/tmp')
    return detect(f'faces/tmp', type=args.type)