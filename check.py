from datasets import load_dataset
import argparse
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# os.system("git clone https://huggingface.co/jyothir/open-math-embeddings")

def math_dataset(split='test'):
    ds = load_dataset('competition_math')[split]
    id2example = {}
    for i in range(len(ds)):
        example = ds[i]
        id2example[i] = {
            'id': i,
            'input': example['problem'],
            'output': example['solution'],
            'meta': {
                'level': example['level'],
                'type': example['type']
            }
        }
    return id2example
def gsm8k(split='test'):
    ds = load_dataset('gsm8k', 'main')[split]
    id2example = {}
    for i in range(len(ds)):
        example = ds[i]
        id2example[i] = {
            'id': i,
            'input': example['question'],
            'output': example['answer'],
            'meta': {
                'example': example
            }
        }
    return id2example

def load_test_dataset(dataset):
    if dataset == 'MATH':
        id2example = math_dataset()
    elif dataset == 'gsm8k':
        id2example = gsm8k()
    elif dataset == 'phrases':
        id2example = phrases()
    else:
        raise NotImplementedError()
    return id2example




def main(args):
    bleu_scores = [];chunk_size = 100
    # Load test dataset and make an index of n-grams
    hf_dataset = load_dataset("keirp/open-web-math-dev")
    test_ds = load_test_dataset(args.test_dataset)
    model = SentenceTransformer(args.encoder_model, device=device);print("model loaded",device)
        # Iterate over the test set in chunks of 100
    with open("open-math-embeddings/embeddings_open.pkl", "rb") as f:
       embeddings = pickle.load(f)

    corpus_embeddings = embeddings
    for start in range(0, len(test_ds), chunk_size):
        end = min(start + chunk_size, len(test_ds))
        chunk = []
        for idx in range(start, end):
            chunk.append(test_ds[idx])
        if args.query_type == 'Q':
            query_embeddings = model.encode([entry['input'] for entry in chunk])
        elif  args.query_type == 'Q&A':
            query_embeddings = model.encode([entry['input']+'\n'+entry['output'] for entry in chunk])
        cos_similarities = cosine_similarity(query_embeddings, corpus_embeddings)

        for i, data in enumerate(chunk):
            query = data['input']
            sim_scores = cos_similarities[i]
            top_3_indices = np.argsort(sim_scores)[::-1][:3]
                
            query_bleu_scores = []  # To store the BLEU scores for the three similar sentences

            for j, index in enumerate(top_3_indices):
                similar_sentence = hf_dataset['train'][int(index)]['text']
                
                reference = [query.split()]
                candidate = similar_sentence.split()
                bleu_score = sentence_bleu(reference, candidate[:500])
                query_bleu_scores.append(bleu_score)  # Append the BLEU score for this sentence

            # Get the maximum BLEU score for this query and store
            # max_bleu_score = max(query_bleu_scores)
            bleu_scores.append(bleu_score);
    with open("bleu_scores.pkl", "wb") as f:
        pickle.dump(bleu_scores, f)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dataset', type=str, default='MATH', choices=['MATH', 'gsm8k', 'phrases'])
    parser.add_argument('--encoder-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--query-type', type=str, default='Q',choices=['Q','Q&A'])
    args = parser.parse_args()
    main(args)


