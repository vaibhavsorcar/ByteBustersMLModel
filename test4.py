    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    import os
    from transformers import MarianMTModel, MarianTokenizer
    # from sacremoses import MosesTokenizer, MosesDetokenizer
    
    # Define your CSV file path
    csv_file_path = 'C:\\Users\\91998\\Downloads\\test.csv'
    
    # Read CSV file using Pandas
    df = pd.read_csv(csv_file_path)
    
    # Extract input and target columns (limit to the first 100 entries)
    input_texts = df['text'][:100]
    target_texts = df['sentiment'][:100]
    
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
    
    # Translate texts
    inputs = tokenizer(input_texts.tolist(), return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    
    # Decode and print the translated text in Hindi
    translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, (input_text, translated_text) in enumerate(zip(input_texts, translated_texts)):
        print(f"Input Text ({i + 1}): {input_text}")
        print(f"Translated Text ({i + 1}): {translated_text}")
        print("-------------------------")
