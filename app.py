import os
from flask import Flask, request, jsonify, send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from huggingface_hub import login




# Deep learning

login('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')
# login(os.environ['API_P9_HUGG'])

MODEL_NAME = 'Rapacki/T5-small-tweet-p9'

# MODEL_NAME = 'T5-small'  # Le chemin vers votre modèle préentraîné
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


# Définir le périphérique (GPU si disponible, sinon CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
device = torch.long
model.to(device)
# model = model.half().to(device)
# Créer l'application Flask
# Désactive les dropout out test 16-10-20:34
model.eval()

app = Flask(__name__)
@app.route("/")
def home():
    return "Hello, welcome to the sentiment classification API for project 09 !"


# Définir un point d'entrée pour la prédiction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    input_text = data.get('text', '')

    # Ajouter le préfixe "sentiment: " à l'entrée pour correspondre au fine-tuning
    input_text = "sentiment: " + input_text

    # Tokenizer l'entrée
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    # Envoyer les tenseurs à l'appareil
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Générer la prédiction
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)

    # Décoder la prédiction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Retourner la réponse JSON
    return jsonify({'sentiment': prediction})

# # Démarrer le serveur Flask
# if __name__ == "__main__":

#     # Launch the Flask app
#     app.run(debug=True)