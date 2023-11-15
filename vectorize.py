import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def save_vectorized_data(vectorized_data, vectorizer, file_path):
    with open(file_path, 'wb') as file:
        data = {
            'vectorized_data': vectorized_data,
            'vectorizer': vectorizer
        }
        pickle.dump(data, file)

def vectorize_data(data):
    tf = TfidfVectorizer(max_features=200, stop_words='english')
    vector = tf.fit_transform(data).toarray()
    return vector, tf 

def main():
    df = pd.read_csv('/home/arnav/capstone_project/imp_proj/data/Tags_Data.csv')
    vector, vectorizer = vectorize_data(df['Tags'])
    save_vectorized_data(vector, vectorizer, '/home/arnav/capstone_project/imp_proj/data/vectorized_data.pkl')

if __name__ == "__main__":
    main()
