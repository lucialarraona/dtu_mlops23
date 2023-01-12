from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from pydantic import BaseModel
from typing import List
import regex as re 

app = FastAPI()

class Item(BaseModel):
    email: str
    domain_match: str


@app.post("/text_model")
def contains_email(data: Item):
   regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
   regex_domain = r'@(\w+)'
   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
      "is_email": re.fullmatch(regex, data.email) is not None,
      "match_domain": re.fullmatch(regex_domain,data.domain_match) is not None
   }
   return response


#@app.post("/items/")
#async def create_item(data: Item):
#    return data