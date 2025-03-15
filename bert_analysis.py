from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


data_pd = pd.read_csv('Codage_CV versus HR_for Irene Berezin-Dec-17-2024.csv')
data_pd = data_pd[data_pd['ResponseM'].notnull()]


classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

def detect_language_transformer(text):
    """
    Given a list of texts, returns their langauge, which is one of 'en', 'fr'.
    """
    result = classifier(text[:512], return_all_scores=True)  # Get all probabilities
    scores = {x['label']: x['score'] for x in result[0]}

    target_languages = ['fr', 'en']
    filtered_scores = {k: scores[k] for k in target_languages if k in scores}
    
    # model was labelling some french text as italian, so forcing it to label it as french
    if filtered_scores:
        return max(filtered_scores, key=filtered_scores.get)
    else:
        # Fallback: If neither 'fr' nor 'en' is detected, choose based on closest match
        if 'it' in scores and scores['it'] >= 0.3:
            return 'fr' 
        return 'en' 

data_pd['language'] = data_pd['ResponseM'].dropna().apply(detect_language_transformer)

label_map = {
    "LABEL_0": "negative",  
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative", 
    "neutral":  "neutral",
    "positive": "positive"
}

english_data = data_pd[data_pd['language'] == 'en']['ResponseM'].astype(str).to_list()
french_data = data_pd[data_pd['language'] == 'fr']['ResponseM'].astype(str).to_list()

batch_size = 32
results_en = []
results_fr = []

def get_sentiment_scores(texts, tokenizer, model, alpha=2.0, batch_size=32):
    """
    Given a collection of texts, a tokenizer, and a model, return the the classified text labels as a list.
    """
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        raw_probs = F.softmax(logits, dim=-1)
        
        adjusted_probs = torch.pow(raw_probs, alpha)
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
        for probs in adjusted_probs:
            # Assuming the model output is [neg, neu, pos]
            all_results.append({
                "negative": probs[0].item(),
                "neutral":  probs[1].item(),
                "positive": probs[2].item()
            })
    return all_results

model_name_en = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
model_en = AutoModelForSequenceClassification.from_pretrained(model_name_en)
model_en.eval() 

model_name_fr = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer_fr = AutoTokenizer.from_pretrained(model_name_fr)
model_fr = AutoModelForSequenceClassification.from_pretrained(model_name_fr)
model_fr.eval()

english_data = data_pd[data_pd['language'] == 'en']['ResponseM'].astype(str).to_list()
french_data  = data_pd[data_pd['language'] == 'fr']['ResponseM'].astype(str).to_list()

results_en = get_sentiment_scores(english_data, tokenizer_en, model_en, batch_size=32)
results_fr = get_sentiment_scores(french_data, tokenizer_fr, model_fr,batch_size=32)

english_mask = (data_pd['language'] == 'en')
french_mask  = (data_pd['language'] == 'fr')

data_pd.loc[english_mask, 'negative'] = [r["negative"] for r in results_en]
data_pd.loc[english_mask, 'neutral']  = [r["neutral"]  for r in results_en]
data_pd.loc[english_mask, 'positive'] = [r["positive"] for r in results_en]

data_pd.loc[french_mask, 'negative'] = [r["negative"] for r in results_fr]
data_pd.loc[french_mask, 'neutral']  = [r["neutral"]  for r in results_fr]
data_pd.loc[french_mask, 'positive'] = [r["positive"] for r in results_fr]

data_pd['sentiment'] = ''

data_pd.loc[(data_pd['negative'] >= data_pd['neutral']) & (data_pd['negative'] >= data_pd['positive']), 'sentiment'] = 'negative'
data_pd.loc[(data_pd['neutral'] > data_pd['negative']) & (data_pd['neutral'] >= data_pd['positive']), 'sentiment'] = 'neutral'
data_pd.loc[(data_pd['positive'] > data_pd['negative']) & (data_pd['positive'] > data_pd['neutral']), 'sentiment'] = 'positive'



import numpy as np

threshold = 0.2

data_pd = data_pd.dropna(subset=['negative', 'neutral', 'positive']).reset_index(drop=True)

def is_uncertain(row, threshold):
    values = [row['negative'], row['neutral'], row['positive']]
    sorted_values = sorted(values, reverse=True)
    return (sorted_values[0] - sorted_values[1]) < threshold

data_pd['is_uncertain'] = data_pd.apply(lambda row: is_uncertain(row, threshold), axis=1)

data_true = data_pd[data_pd['is_uncertain'] == True]
print(len(data_true))

data_false = data_pd
print(len(data_false))