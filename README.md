Machine Learning Projects
1. Sentiment Analysis with TensorFlow
Project Overview
This Python script demonstrates sentiment analysis using TensorFlow.
It employs a neural network model to classify sentiments in a given dataset. Key features include:

Input data preprocessing using Pandas.
Tokenization and padding of input sequences.
Model training and evaluation for sentiment analysis.
Usage

Install Dependencies:
pip install tensorflow pandas

Train a sentiment analysis model and evaluates its performance.


2. English-to-Hindi Machine Translation with TensorFlow and Hugging Face Transformers
Project Overview
This Python script showcases machine translation using TensorFlow and Hugging Face Transformers. It translates English texts to Hindi using
a pre-trained model (Helsinki-NLP/opus-mt-en-hi). Key features include:

Integration with the Hugging Face Transformers library.
Tokenization and padding of input sequences.
Translation of English texts to Hindi.
Requirements
TensorFlow , Pandas , Hugging Face Transformers.

Install Dependencies:
pip install tensorflow pandas transformers

Set CSV File Path:
Update the csv_file_path variable with the path to your CSV file.
Reads a CSV file using Pandas, extracting input and target columns.
Tokenize and Pad Sequences:

Utilizes TensorFlow Tokenizer for data preprocessing.
Load Pre-trained Model and Tokenizer:
Loads a pre-trained English-to-Hindi translation model and tokenizer from the Hugging Face model hub.
Translates input texts to Hindi using the loaded model and tokenizer.
Print Translations:

Decodes and prints the original and translated texts in Hindi.
