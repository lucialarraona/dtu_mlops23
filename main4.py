from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from fastapi import FastAPI, File, Query
#from pydantic.validators import validate_model
from pydantic import validators
#from pydantic import Range


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

def predict_step(image_paths, max_length):
   images = []
   for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
         i_image = i_image.convert(mode="RGB")

      images.append(i_image)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   return preds[0]

app = FastAPI()

@app.post("/predict")
async def predict(image: bytes = File(...), max_length: int = Query(None, gt=1, lt=16)):
    # Convert the bytes to an image file
    with open("image.jpg", "wb") as f:
        f.write(image)
    # Pass the image path and max_length to the predict_step function
    caption = predict_step(["image.jpg"], max_length= int(max_length))
    return {"caption": caption}



