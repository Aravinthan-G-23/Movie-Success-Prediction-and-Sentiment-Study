# Movie-Success-Prediction-and-Sentiment-Study
project :-Movie Success Prediction and Sentiment Study
import pandas as pd

# Load your IMDB/Kaggle data
df = pd.read_csv('imdb_movie_dataset.csv')

# Preview data
print(df.head())

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
numeric_features = ['Year', 'Runtime (Minutes)', 'Rating', 'Votes', 'Metascore']
categorical_features = ['Genre']

# Encode 'Genre'
encoder = OneHotEncoder()
genre_encoded = encoder.fit_transform(df[['Genre']].fillna('')).toarray()
genre_labels = encoder.get_feature_names_out(['Genre'])

# Construct features DataFrame
X_numeric = df[numeric_features].fillna(0)
X = np.hstack((X_numeric, genre_encoded))
y = df['Revenue (Millions)'].fillna(0).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# RMSE calculation
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print('Test set RMSE:', rmse)
print('Test set R2:', r2_score(y_test, predictions))

import nltk
import sys
import os

# Suppress NLTK download messages
nltk.download('vader_lexicon', quiet=True)



import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Dummy DataFrame 'reviews_df' with columns: 'movie_title', 'review_text'
# reviews_df = pd.read_csv('movie_reviews.csv')

# Example only, replace with real review data
reviews_df = pd.DataFrame({
    'movie_title': ['Guardians of the Galaxy', 'The Great Wall'],
    'review_text': [
        'Loved the special effects and humor!',
        'Visuals were great but the plot was weak.'
    ]
})

# Compute sentiments
reviews_df['sentiment'] = reviews_df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Aggregate: Mean sentiment by movie
sentiment_map = reviews_df.groupby('movie_title')['sentiment'].mean().reset_index()
print(sentiment_map)

# Map sentiment to genre
genre_sentiment = []
for _, row in sentiment_map.iterrows():
    genres_str = df[df['Title'] == row['movie_title']]['Genre']
    if not genres_str.empty:
        for genre in genres_str.values[0].split(','):
            genre_sentiment.append({'Genre': genre.strip(), 'Sentiment': row['sentiment']})

genre_sentiment_df = pd.DataFrame(genre_sentiment)
print(genre_sentiment_df.groupby('Genre').Sentiment.mean().sort_values(ascending=False))


import matplotlib.pyplot as plt

avg_sentiment = genre_sentiment_df.groupby('Genre').Sentiment.mean().sort_values()
avg_sentiment.plot(kind='barh', figsize=(8, 6), title='Average Sentiment by Genre')
plt.xlabel('VADER Compound Sentiment')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


