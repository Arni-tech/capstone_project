import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ToolFinder:
    def __init__(self, vectorizer_path, info_csv_path):
        self.vectorizer_path = vectorizer_path
        self.info_csv_path = info_csv_path
        self.vectorizer = self.load_vectorizer()

    def load_vectorizer(self):
        with open(self.vectorizer_path, 'rb') as file:
            data = pickle.load(file)
            return data['vectorizer']

    def find_tool(self, query, num_results=5):
        new_df = pd.read_csv('/home/arnav/capstone_project/imp_proj/data/Tags_Data.csv')
        with open(self.vectorizer_path, 'rb') as file:
            data = pickle.load(file)
            data_vectors = data['vectorized_data']

        query_vector = self.vectorizer.transform([query.lower()]).toarray()
        similarities = cosine_similarity(query_vector, data_vectors)

        most_similar_indices = np.argsort(similarities[0])[::-1][:num_results]
        most_similar_tool_names = [new_df.iloc[index]['Tool Name'] for index in most_similar_indices]

        return most_similar_tool_names

    def find_info(self, tool_names):
        df_info = pd.read_csv(self.info_csv_path)
        tool_information = df_info[df_info['Tool Name'].isin(tool_names)]
        tool_info_dict = {row['Tool Name']: row for _, row in tool_information.iterrows()}
        ordered_tool_info = [tool_info_dict[tool_name] for tool_name in tool_names]

        return ordered_tool_info

    def main(self):
        user_query = "give me a leads management tool"
        top_similar_tools = self.find_tool(user_query, num_results=5)
        tool_info = self.find_info(top_similar_tools)

        print("Top similar tools:", top_similar_tools)
        print("Tool information:", tool_info)

        

if __name__ == "__main__":
    tool_finder = ToolFinder('/home/arnav/capstone_project/imp_proj/data/vectorized_data.pkl',
                             '/home/arnav/capstone_project/imp_proj/data/usable_dataset1.csv')
    tool_finder.main()
