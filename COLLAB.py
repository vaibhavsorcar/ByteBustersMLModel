import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import sentencepiece
from transformers import MarianMTModel, MarianTokenizer ,AdamW

# Define your CSV file path
csv_file_path = 'C:\\Users\\91998\\Downloads\\test.csv'

# Read CSV file using Pandas
df = pd.read_csv(csv_file_path)
print(df)

# Extract input and target columns
input_texts = df['text']
target_texts = df['sentiment']

# Define other parameters
batch_size = 32
max_length_input = 100
max_length_target = 100
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
embedding_dim = 50  # Define the embedding dimension
epochs = 10  # Define the number of epochs

# Initialize tokenizers
input_tokenizer = Tokenizer(oov_token=oov_token)
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer = Tokenizer(oov_token=oov_token)
target_tokenizer.fit_on_texts(target_texts)

# Get vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Tokenize and pad sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_length_input, padding=padding_type, truncating=trunc_type)

target_sequences = target_tokenizer.texts_to_sequences(target_texts)
padded_target_sequences = pad_sequences(target_sequences, maxlen=max_length_target, padding=padding_type, truncating=trunc_type)

# Load pre-trained English-to-Hindi translation model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Translate new texts
new_texts = ["Hello How are you", "what is your name", "who are you","why are you so tall"]
input_sequences = input_tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(input_sequences, maxlen=max_length_input, padding=padding_type, truncating=trunc_type)

# Tokenize and translate
inputs = tokenizer(new_texts, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)

# Decode and print the translated text in Hindi
translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, (input_text, translated_text) in enumerate(zip(new_texts, translated_texts)):
    print(f"Input Text ({i + 1}): {input_text}")
    print(f"Translated Text ({i + 1}): {translated_text}")
    print("-------------------------")
