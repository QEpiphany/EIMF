import re
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('C:/Code/MyModel/LLMs/bert-base-uncased')
model = BertModel.from_pretrained("C:/Code/MyModel/LLMs/bert-base-uncased")


def text_to_embedding(text):
    max_chunk_len = 200
    if (len(text) > max_chunk_len):
        text_chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        chunks_embeddings = []
        for chunk in text_chunks:
            inputs = tokenizer.encode(chunk, return_tensors="pt")
            with torch.no_grad():
                outputs = model(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            chunks_embeddings.append(embeddings)
        final_encoding = torch.cat(chunks_embeddings, dim=1)
        return final_encoding
    else:
        tokens = tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

def spilit_text(seq):
    values = re.findall(r':\s*"([^"]*)"', seq)
    embeddings_list = []
    for value in values[:20]:
        value_emb = text_to_embedding(value)
        value_emb = unify_second_dimension(value_emb)
        embeddings_list.append(value_emb)

    final_embeddings = torch.stack(embeddings_list, dim=0).squeeze(1)
    return final_embeddings


def unify_second_dimension(embedding):
    target_dim = 100
    if embedding.size(1) != target_dim:
        linear_layer = torch.nn.Linear(embedding.size(1), target_dim)
        processed_embedding = linear_layer(embedding)
    else:
        processed_embedding = embedding.clone()
    return processed_embedding


if __name__ == '__main__':
    dataset = "Beauty"
    phrase = "train"

    text_path = dataset + '/Prompts/typical_texts_with_responses_{}_5.xlsx'.format(phrase) #Beauty xlsx

    df = pd.read_excel(text_path)

    df['Text_Embedding'] = None

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Response']
        text_emb = spilit_text(text)
        df.at[i, 'Text_Embedding'] = str(text_emb.tolist())

    df.to_excel(dataset + '/Prompts/typical_texts_emb_{}_5.xlsx'.format(phrase), index=False)


