# ================= BART MODEL =================
from transformers import BartTokenizer, BartForConditionalGeneration

bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

def bart_summarize(text):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ================= BERT MODEL (Extractive) =================
import numpy as np
import nltk
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

# Download once
nltk.download('punkt')

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_sentence_embedding(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

def create_embeddings(sentences):
    embeddings = [get_sentence_embedding(s) for s in sentences]
    return np.vstack(embeddings)

def build_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def rank_sentences(similarity_matrix):
    scores = similarity_matrix.sum(axis=1)
    return np.argsort(-scores)

def bert_summarize(text, num_sentences=2):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    embeddings = create_embeddings(sentences)
    sim_matrix = build_similarity_matrix(embeddings)
    ranked = rank_sentences(sim_matrix)

    selected = sorted(ranked[:num_sentences])
    summary = [sentences[i] for i in selected]

    return " ".join(summary)