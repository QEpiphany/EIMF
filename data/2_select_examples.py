import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AffinityPropagation, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
def text_to_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


def cluster_texts_with_bert(texts_dict, damping=0.8, max_iter=500, convergence_iter=15):
    """
    Uses BERT and AffinityPropagation to cluster texts.

    Parameters:
    texts_dict (dict): Dictionary where keys are identifiers and values are lists of texts.
    damping (float): Damping factor for AffinityPropagation, default is 0.8.
    max_iter (int): Maximum number of iterations, default is 500.
    convergence_iter (int): Number of iterations required for convergence, default is 15.

    Returns:
    clusters (dict): Clustering result, where keys are cluster centers and values are lists of keys belonging to each cluster.
    """

    tokenizer, model = load_bert_model()
    keys = list(texts_dict.keys())
    embeddings = []

    # Convert each key's corresponding text list into BERT vector representations
    for key in tqdm(keys, desc="Processing texts"):
        text_list = texts_dict[key]
        text_embeddings = [text_to_bert_embedding(text, tokenizer, model) for text in text_list]
        flattened_embedding = np.mean(np.array(text_embeddings), axis=0).reshape(1, -1)
        embeddings.append(flattened_embedding)

    # Stack all samples into a single numpy array
    embeddings = np.concatenate(embeddings, axis=0)

    # Perform clustering using AffinityPropagation
    af = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, preference=-5).fit(embeddings) #Office -5, Beauty|Grocery -10
    labels = af.labels_
    cluster_centers_indices = af.cluster_centers_indices_

    # Visualize the clustering results
    for cluster in np.unique(labels):
        row_ix = np.where(labels == cluster)
        plt.scatter(embeddings[row_ix, 0], embeddings[row_ix, 1])

    plt.show()

    # Construct the clustering result dictionary
    clusters = {}
    cluster_centers_dict = {}

    # Calculate the distance matrix between all points
    all_distances = squareform(pdist(embeddings))

    for i, cluster_label in enumerate(labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(keys[i])

        if cluster_label not in cluster_centers_dict:
            if cluster_label in cluster_centers_indices:
                cluster_centers_dict[cluster_label] = keys[cluster_centers_indices[cluster_label]]
            else:
                # Find the point closest to the cluster center
                center = af.cluster_centers_[cluster_label]
                distances = np.linalg.norm(embeddings - center, axis=1)
                closest_index = np.argmin(distances)
                cluster_centers_dict[cluster_label] = keys[closest_index]

    return clusters, cluster_centers_dict


def load_bert_model():
    model_name = "../PLMs/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def create_prompts(sequence_texts):
    original_text = ("The user's historical click sequence is as follows: {}, ".format(sequence_texts) +
                     "please infer the user's interest preference and output it in the format of json, such as {interest sequence number: interest content;}")
    return original_text


def main():
    dataset = 'Grocery' # Office_Products | Grocery | Book | Beauty
    file_name = 'train' # train | test |valid
    data_file = dataset + '/{}_{}.txt'.format(dataset, file_name)
    name_file = dataset + '/{}_item_name.csv'.format(dataset)

    data_df = pd.read_csv(data_file, sep=',', header=None, names=['user_id', 'item_id', 'seq_id', 'time'])


    sorted_items = data_df.sort_values(['user_id', 'seq_id']).groupby('user_id')['item_id']

    data_dict = {user_id: list(items) for user_id, items in sorted_items}


    id2name_df = pd.read_csv(name_file)
    id2name_dict = {item_id: item_name for item_id, item_name in id2name_df.values}
    for user_id, item_id_list in data_dict.items():
        name_list = [str(id2name_dict[item_id]) for item_id in item_id_list if item_id in id2name_dict]
        data_dict[user_id] = name_list


    keys = list(data_dict.keys())
    half_point = len(keys) // 20

    # first_half_keys = keys[:half_point]
    # first_half_dict = {k: data_dict[k] for k in first_half_keys}

    clusters, cluster_centers = cluster_texts_with_bert(data_dict)

    result_df = pd.DataFrame(columns=['ID', 'Cluster_label', 'Cluster_Center_Index'])

    print(len(clusters))
    for cluster_id, id_list in clusters.items():
        center_index = cluster_centers[cluster_id]
        for id in id_list:
            result_df = result_df.append({'ID': id, 'Original Text': data_dict[id], 'Cluster_label': cluster_id, 'Cluster_Center_Index': center_index}, ignore_index=True)
    print(len(result_df))
    typical_df = pd.DataFrame(columns=['Cluster_Center_Index', 'Typical Text'])
    for cluster_id, center_index in cluster_centers.items():
        center_index = cluster_centers[cluster_id]
        center_text = data_dict[center_index]
        center_text = create_prompts(center_text)
        typical_df = typical_df.append({'Cluster_Center_Index': center_index, 'Typical Text': center_text}, ignore_index=True)
    result_df.to_csv('{}/Prompts/'.format(dataset)+'clusters_with_centers_all_{}_5.csv'.format(file_name), index=False)
    typical_df.to_csv('{}/Prompts/'.format(dataset)+'typical_texts_all_{}_5.csv'.format(file_name), index=False)



if __name__ == '__main__':
    main()