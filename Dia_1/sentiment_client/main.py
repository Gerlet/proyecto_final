from sentiment_analyzer.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

text = "El producto es excelente, lo recomiendo mucho."
predict_text = analyzer.predict(text)

print(f"Texto: {text}")
print(f"Clasificación: {predict_text}")