from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('sentiment_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
max_sequence_length = 52
# Mapping dictionary
sentiment_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

# Function to preprocess user input
def preprocess_input(user_input, tokenizer, max_sequence_length):
    # Tokenize and pad the input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(input_sequence, maxlen=max_sequence_length)
    return padded_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    
    # Preprocess the user input
    preprocessed_input = preprocess_input(user_input, tokenizer, max_sequence_length)
    
    # Make predictions using the trained model
    predictions = model.predict(preprocessed_input)
    
    # Convert predictions to sentiment labels using the mapping
    predicted_sentiment_label = sentiment_mapping[np.argmax(predictions)]
    
    return render_template('index.html', user_input=user_input, predicted_sentiment=predicted_sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
