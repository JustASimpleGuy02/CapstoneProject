import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from functions.search import search_image

app = FastAPI()

class TextInput(BaseModel):
    
    text: str
    
@app.post("/search")
def search_item(input_data: TextInput):
    processed_text = input_data.text.lower()
    json_obj = search_image(prompt=processed_text)
    return json_obj


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)