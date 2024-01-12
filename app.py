from flask import Flask, render_template, request
import pandas as pd
from predict import load_model
from predict import get_sentiment

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    df = pd.DataFrame([form_data])
    
    df['type'] = df['type'].map({'MOVIE': 0, 'SHOW': 1})

    df['age_certification'] = df['age_certification'].astype('category')

    df['description_sentiment'] = df['description'].apply(get_sentiment)

    df['title_sentiment'] = df['title'].apply(get_sentiment)

    df['description_length'] = df['description'].apply(len)
    
    df['title_length'] = df['description'].apply(len)

    X_df = df[['release_year', 'runtime', 'type', 'description_sentiment', 'description_length', 'title_sentiment', 'title_length']]

    user_prediction = model.predict(X_df)


    return render_template('index.html', prediction=user_prediction[0], user_input=df)
if __name__ == '__main__':
    app.run(debug=True)
