from fastapi import UploadFile, File
from typing import Optional
from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from pydantic import BaseModel
from typing import List
import regex as re 
import cv2
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h = 500, w = 500):
   with open('image.jpg', 'wb') as image:
      content = await data.read()
      image.write(content)
      image.close()
      img = cv2.imread("image.jpg")
      res = cv2.resize(img, (int(h), int(w)))
      cv2.imwrite('image_resize.jpg', res)

   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
   }
   return FileResponse('image_resize.jpg')