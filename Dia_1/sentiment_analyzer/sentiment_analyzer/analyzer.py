import os
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification


class SentimentAnalyzer:
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__),'../sentiment_model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self,text):
        # Determinar el dispositivo (GPU si está disponible, de lo contrario CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mover el modelo al dispositivo
        model = self.model.to(device)

        # Tokenizar el texto
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Mover los tensores al mismo dispositivo que el modelo
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Pasar inputs al modelo
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits

        # Calcular probabilidades usando softmax
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        prediction = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][prediction]

        predict_text = 'neutro'

        if confidence >= 0.5:
            predict_text = 'positivo'
        elif confidence <= 0:
            predict_text = 'negativo'

        return predict_text