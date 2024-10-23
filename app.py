# 19/10/2024 - API avec interprétation SHAP. A MODIFIER AVEC GRAPH interprétation puis DEPLOYER

import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import shap
from huggingface_hub import login


app = Flask(__name__)
# Deep learning

login('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')
# login(os.environ['API_P9_HUGG'])

MODEL_NAME = 'Rapacki/T5-small-tweet-p9'

# MODEL_NAME = 'T5-small'  # Le chemin vers votre modèle préentraîné
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


# Définir le périphérique (GPU si disponible, sinon CPU)

device = torch.device("cpu")

model.to(device)
# model = model.half().to(device)
# Créer l'application Flask
# Désactive les dropout out test 16-10-20:34
model.eval()


@app.route("/")
def home():
    return "Hello, welcome to the sentiment classification API for project 09 !"



def predict_proba(texts):
    """
    Fonction qui prend une liste de textes, les encode,
    et retourne une matrice de probabilités pour les classes "positive" et "negative".
    """
    results = []
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)#test!!!

        # Génération de texte par le modèle T5
        outputs = model.generate(input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculer une probabilité "approximée" pour chaque classe
        if "positive" in generated_text.lower():
            results.append([0.1, 0.9])  # Probabilité plus élevée pour "positive"
        else:
            results.append([0.9, 0.1])  # Probabilité plus élevée pour "negative"

    return np.array(results)




explainer = shap.Explainer(predict_proba, tokenizer)

# Définir un point d'entrée pour la prédiction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    input_text = data.get('text', '')
    
    # Ajouter le préfixe "sentiment: " à l'entrée pour correspondre au fine-tuning
    # input_text = "sentiment: " + input_text
    full_input = f"sentiment: {input_text}"
    # Tokenizer l'entrée
    inputs = tokenizer.encode_plus(
        full_input,
        return_tensors="pt",
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    # Envoyer les tenseurs à l'appareil
    input_ids = inputs['input_ids'].to(device,torch.long)
    attention_mask = inputs['attention_mask'].to(device,torch.long)
    
    # Générer la prédiction
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
        # Décoder la prédiction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "positive" in prediction.lower():
        sentiment = "positive"
        probability = 0.9
    else:
        sentiment = "negative"
        probability = 0.1
    
    

    #Interprétation avec SHAP
    shap_values = explainer([full_input])
    word_contributions = []
    for token, value in zip(shap_values.data[0], shap_values.values[0]):
        if token in {'▁sentiment',':'}:
            continue
        value = value.flatten()[0]
        word_contributions.append({'word': token, 'contribution': float(value)})

    # Retourner la prédiction et l'interprétation
    return jsonify({
        'sentiment': sentiment,
        'interpretation': word_contributions
    })
    

# Démarrer le serveur Flask
# if __name__ == "__main__":

#     # Launch the Flask app
#     app.run(debug=True)