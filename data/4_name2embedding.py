import re
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('C:/Code/MyModel/LLMs/bert-base-uncased')
model = BertModel.from_pretrained("C:/Code/MyModel/LLMs/bert-base-uncased")

def text_to_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**tokens)

    # Extract embeddings from the output
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def unify_second_dimension(embedding):
    target_dim = 100
    if embedding.size(1) != target_dim:
        linear_layer = torch.nn.Linear(embedding.size(1), target_dim)
        processed_embedding = linear_layer(embedding)
    else:
        processed_embedding = embedding.clone()
    return processed_embedding


if __name__ == '__main__':
    dataset = "Grocery"
    text_path = dataset + '/{}_item_name.csv'.format(dataset)
    df = pd.read_csv(text_path)

    df['Text_Embedding'] = None

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['i_name']
        if type(text) == str:
            text_emb = text_to_embedding(text)
            text_emb = unify_second_dimension(text_emb)
            df.at[i, 'Text_Embedding'] = str(text_emb.tolist())
        else:
            text = str(row['item_id'])
            print(text)
            text_emb = text_to_embedding(text)
            text_emb = unify_second_dimension(text_emb)
            df.at[i, 'Text_Embedding'] = str(text_emb.tolist())

    zero = np.zeros((1, 100))
    zero_list = [zero.tolist()]
    new_row_data = {'item_id': 0,  'i_name':'0', 'Text_Embedding': zero_list}
    new_row = pd.DataFrame(new_row_data, index=[0])

    updated_df = pd.concat([new_row, df], ignore_index=True)

    updated_df.to_csv(dataset + '/{}_item_name_emb_100.csv'.format(dataset), index=False)