from transformers import AutoTokenizer,AutoModelForSequenceClassification

path = './nlp_model'

#cargamos modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)

print("Modelo y Tokenizer cargados...")

import torch
import numpy as np

# Función para predecir si un texto es positivo o negativo
def predict_sentiment(text, model, tokenizer):
    # Determinar el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mover el modelo al dispositivo
    model = model.to(device)

    # Tokenizar el texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

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
    print(f'prediction : {probabilities}')
    confidence = probabilities[0][prediction]

    return prediction, confidence

text = "Me encantó la experiencia, fue increíble y lo recomiendo."
pred,conf = predict_sentiment(text,model,tokenizer)
print(f'Texto : {text}')
print(f'prediccion : {pred}')
print(f' confianza : {round(conf)}')
if round(conf) >= 1 :
  print('es positivo')
else:
  print('es negativo')