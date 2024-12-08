from fastapi import FastAPI
from pydantic import BaseModel

from sentiment_analyzer.analyzer import SentimentAnalyzer

app = FastAPI()

class TextInput(BaseModel):
    text :str

@app.get('/')
async def root():
    return {'message':'api de analisis de sentimientos en fastapi'}

@app.post('/sentiment')
async def analyze_text(input: TextInput):
    text = input.text
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.predict(text)

    return {'sentiment':sentiment}