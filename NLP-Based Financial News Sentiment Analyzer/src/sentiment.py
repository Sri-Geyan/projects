from transformers import pipeline

sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model='ProsusAI/finbert'
)

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    score = result['score']

    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:
        return 0.0