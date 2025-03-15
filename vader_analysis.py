from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

data_pd = pd.read_csv('Codage_CV versus HR_for Irene Berezin-Dec-17-2024.csv')
data_pd = data_pd[data_pd['ResponseM'].notnull()]


classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

def detect_language_transformer(text):
    """
    Given a list of strings, returns their language classification, one of 'en' or 'fr'. 
    """
    result = classifier(text[:512], return_all_scores=True)  # Get all probabilities
    scores = {x['label']: x['score'] for x in result[0]} #assigns top probability
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


# Instantiate VADER
vader_analyzer = SentimentIntensityAnalyzer()
# Instantiate the translation pipeline for French to English.
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def translate_to_english(text, lang):
    """
    Given a list of strings and a language, convert text to language.
    """
    if lang == 'fr':
        # The translator returns a list of dicts; extract the translation.
        translation = translator(text, max_length=512)
        return translation[0]['translation_text']
    else:
        return text

data_pd['ResponseM'] = data_pd['ResponseM'].fillna('').astype(str)
data_pd['translated'] = data_pd.apply(lambda row: translate_to_english(row['ResponseM'], row['language']), axis=1) #translate
data_pd['vader'] = data_pd['translated'].apply(lambda text: vader_analyzer.polarity_scores(text)) #apply vader

vader_df = data_pd['vader'].apply(pd.Series)
vader_df = vader_df.rename(columns={
    'neg': 'vader_neg',
    'neu': 'vader_neu',
    'pos': 'vader_pos',
    'compound': 'vader_compound'
})
data_pd = data_pd.join(vader_df).drop(columns=['vader'])

vader_list = data_pd['vader_compound']
sentiment_values = []
for l in vader_list.to_list(): 
    if l < -0.5:
        sentiment_values.append('negative')
    elif l >= -0.5 and  l <= 0.5:
        sentiment_values.append('neutral')
    else: sentiment_values.append('positive')
    
data_pd['vader_sentiment'] = sentiment_values
print(len(data_pd[(data_pd.vader_sentiment == 'neutral') & (data_pd.language == 'fr')])/len(data_pd))
print(data_pd)

def summary_stat(df):
    # Count values for each sentiment category
    neg_vals = len(df[df.vader_sentiment == 'negative'])
    pos_vals = len(df[df.vader_sentiment == 'positive'])
    neu_vals = len(df[df.vader_sentiment == 'neutral'])
    
    # Total number of entries
    total = len(df)
    
    # Prepare dictionary with values and proportions
    sentiment_summary = {
        'neg': neg_vals,
        'pos': pos_vals,
        'neu': neu_vals,
        'prop_neg': neg_vals / total,
        'prop_pos': pos_vals / total,
        'prop_neu': neu_vals / total
    }
    
    # Return as a DataFrame
    return pd.DataFrame([sentiment_summary])

french_df = summary_stat(data_pd[data_pd.language == 'fr'])
english_df = summary_stat(data_pd[data_pd.language == 'en'])
total_df = summary_stat(data_pd)

print(french_df)
print(english_df)
print(total_df)