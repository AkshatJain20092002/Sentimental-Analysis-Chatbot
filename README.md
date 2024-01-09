# Sentimental Analysis Chatbot

A simple chatbot that performs sentiment analysis on user input using a trained Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) for Twitter Data.

## Datasets

1. [Twitter and Reddit Sentiment Analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)

2. [Twitter 1.6 Million Tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)


## Overview

This project implements a sentiment analysis chatbot that predicts the sentiment (positive, negative, or neutral) of user input. The chatbot is built using natural language processing techniques, a machine learning model, and a web interface for user interaction.

## Features

- Sentiment analysis using a trained RNN-LSTM model.
- Web-based chatbot interface built with Flask.
- Preprocessing of user input, including lowercasing, removing special characters, and tokenization.
- TF-IDF feature extraction for numerical representation of text data.
- Integration with a simple web interface for easy user interaction.

## Usage
Clone the repository:

   ```bash
   git clone https://github.com/AkshatJain20092002/sentimental-analysis-chatbot.git
   cd sentimental-analysis-chatbot

## Project Structure

├── app.py               

├── templates/           

│   └── index.html

├── static/css/
              
│   └── style.css

├── tokenizer.pickle

├── sentiment_model.h5   (For this run the .ipynb file)

## License
This project is licensed under the MIT License .
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



