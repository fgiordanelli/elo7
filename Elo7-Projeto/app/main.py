from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
from typing import List


model = pickle.load(open(r'app/elo7_category.sav', 'rb'))

app = FastAPI()

class Product(BaseModel):
    title: str

class ItemIn(BaseModel):
    products: List[Product]

class ItemOut:
    def __init__(self, c):
        self.categories=c
    categories: List[str]

@app.post('/')
def predict(items: ItemIn): 
    looping = list(map(lambda x : x.title, items.products))
    predict = model.predict(pd.Series(looping))
    predict_json = pd.Series(predict).to_json(orient='values')
    predict_output = ItemOut(predict_json)
    return predict_output
