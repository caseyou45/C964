import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob


# Helper Function to get sentiment 
def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

def load_model():
    #Pull in csv data
    df = pd.read_csv('netflix.csv')

    # Fill missing values in numeric columns with their mean
    numeric_columns = ['release_year', 'runtime', 'imdb_score']
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Convert 'age_certification' to categorical and fill missing values with mode
    df['age_certification'] = df['age_certification'].astype('category')
    df['age_certification'].fillna(df['age_certification'].mode()[0], inplace=True)

    # Convert 'type' to a numeric value (0 for 'MOVIE', 1 for 'SHOW') and also fill missing with mode 
    df['type'] = df['type'].map({'MOVIE': 0, 'SHOW': 1})
    df['type'].fillna(df['type'].mode()[0], inplace=True)

    #Fill missing for 'description', get 'description_sentiment', and 'description_length'
    df['description'].fillna('', inplace=True)
    df['description_sentiment'] = df['description'].apply(get_sentiment)
    df['description_length'] = df['description'].apply(len)

    #Same process above for description is applied to title; produces 'title_sentiment' and 'title_length'
    df['title'].fillna('', inplace=True)
    df['title_sentiment'] = df['title'].apply(get_sentiment)
    df['title_length'] = df['title'].apply(len)

    #Make the split and train the model 
    X = df[['release_year', 'runtime', 'type', 'description_sentiment', 'description_length', 'title_sentiment', 'title_length']]
    y = df['imdb_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    test(model, X_train, X_test, y_train, y_test)


    return model


def test(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    
    
    # test_movie = {
    # 'release_year': 1994,
    # 'runtime': 135,
    # 'type': 0, 
    # 'description': 'A prominent banker unjustly convicted of murder spends many years in the Shawshank prison. He is befriended by a convict who knows the ropes and helps him to cope with the frightening realities of prison life.',
    # 'title': 'The Shawshank Redemption',
    # 'age_certification' : 'R',
    
    # }
    
    test_movie = {
    'release_year': 2017,
    'runtime': 95,
    'type': 0, 
    'description': 'A boy haunted by visions of a dark tower from a parallel reality teams up with the towers disillusioned guardian to stop an evil warlock known as the Man in Black who plans to use the boy to destroy the tower and open the gates of Hell.',
    'title': 'The Dark Tower',
    'age_certification' : 'R',
    
    }
    
    test_df = pd.DataFrame([test_movie])

    test_df['age_certification'] = test_df['age_certification'].astype('category')

    test_df['description_sentiment'] = test_df['description'].apply(get_sentiment)

    test_df['title_sentiment'] = test_df['title'].apply(get_sentiment)

    test_df['description_length'] = test_df['description'].apply(len)
    
    test_df['title_length'] = test_df['title'].apply(len)

    X_test_df = test_df[['release_year', 'runtime', 'type', 'description_sentiment', 'description_length', 'title_sentiment', 'title_length']]

    test_prediction = model.predict(X_test_df)

    print(f'{test_movie["title"]} Prediction:')
    print(f'IMDb Score: {test_prediction[0]}')