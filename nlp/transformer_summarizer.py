import torch
import numpy as np
import nltk

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BertTokenizer,
    BertModel
)

from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

# =========================
# DOWNLOAD REQUIRED NLTK DATA
# =========================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# =========================
# DEVICE CONFIGURATION
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# BART MODEL SETUP
# =========================

BART_MODEL_NAME = "facebook/bart-large-cnn"

bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)

bart_model = BartForConditionalGeneration.from_pretrained(
    BART_MODEL_NAME
).to(device)

bart_model.eval()


# =========================
# BART SUMMARIZATION
# =========================

def bart_summarize(text):

    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."

    try:

        inputs = bart_tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(device)

        with torch.no_grad():

            summary_ids = bart_model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = bart_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary

    except Exception as e:
        return f"BART summarization error: {str(e)}"


# =========================
# BERT MODEL SETUP
# =========================

BERT_MODEL_NAME = "bert-base-uncased"

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

bert_model = BertModel.from_pretrained(
    BERT_MODEL_NAME
)

bert_model.eval()


# =========================
# SENTENCE EMBEDDING
# =========================

def get_sentence_embedding(sentence):

    inputs = bert_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.detach().numpy()


# =========================
# CREATE EMBEDDINGS
# =========================

def create_embeddings(sentences):

    embeddings = [
        get_sentence_embedding(sentence)
        for sentence in sentences
    ]

    return np.vstack(embeddings)


# =========================
# BUILD SIMILARITY MATRIX
# =========================

def build_similarity_matrix(embeddings):

    return cosine_similarity(embeddings)


# =========================
# RANK SENTENCES
# =========================

def rank_sentences(similarity_matrix):

    scores = similarity_matrix.sum(axis=1)

    return np.argsort(-scores)


# =========================
# BERT SUMMARIZATION
# =========================

def bert_summarize(text, num_sentences=2):

    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."

    try:

        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        embeddings = create_embeddings(sentences)

        similarity_matrix = build_similarity_matrix(
            embeddings
        )

        ranked_sentences = rank_sentences(
            similarity_matrix
        )

        selected_sentences = sorted(
            ranked_sentences[:num_sentences]
        )

        summary = [
            sentences[i]
            for i in selected_sentences
        ]

        return " ".join(summary)

    except Exception as e:
        return f"BERT summarization error: {str(e)}"