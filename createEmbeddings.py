import pandas as pd
import tiktoken
import openai


from openai.embeddings_utils import get_embedding

openai.api_key = "INSERT YOUR API KEY HERE"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# load & inspect dataset
input_datapath = "data/all_in_tweets.csv"
df = pd.read_csv(input_datapath)

print(df.columns)
df = df[['Name', 'Tweet']]
df = df.dropna()

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000
df = df.sort_index().tail(top_n * 2)

# Combine "Supplier" and "Description" columns into a single "combined" column
df['combined'] = df['Name'] + ' ' + df['Tweet']

encoding = tiktoken.get_encoding(embedding_encoding)


# This may take a few minutes
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("data/all_in_tweets_embeddings.csv")
