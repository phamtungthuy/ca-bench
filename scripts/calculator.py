import string
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from enum import Enum
import pandas as pd
from openai import OpenAI

from utils.constants import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL_NAME

from codebleu import calc_codebleu
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu

import numpy as np


class MetricType(Enum):
    ACCURACY = "accuracy"
    NUMERICAL_ACCURACY = "numerical_accuracy"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    WORD_SIMILARITY = "word_similarity"
    CODE_BLEU = "code_bleu"
    BLEU = "bleu"
    F1 = "f1"
    R2 = "r2"


class Calculator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.client = OpenAI(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

    def get_openai_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Get the embedding vector for a given text using the OpenAI API and a specified model.
        """
        
            
        response = self.client.embeddings.create(input=[text], model=model)
        embedding = np.array(response.data[0].embedding)
        
        return embedding

    def get_openai_embeddings_batch(
        self, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple texts in batches for better performance.
        Returns a dictionary mapping text to embedding.
        """
        results = {}
        
        # Check what's already cached
        # Fetch remaining texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if batch:  # Only make API call if there are texts to fetch
                try:
                    response = self.client.embeddings.create(input=batch, model=model)
                    for j, embedding_data in enumerate(response.data):
                        text = batch[j]
                        embedding = np.array(embedding_data.embedding)
                        
                        results[text] = embedding
                except Exception as e:
                    print(f"Error getting embeddings for batch: {e}")
                    # Fallback to individual calls for this batch
                    for text in batch:
                        try:
                            results[text] = self.get_openai_embedding(text, model)
                        except Exception as e2:
                            print(f"Error getting embedding for '{text}': {e2}")
                            # Use zero vector as fallback
                            results[text] = np.zeros(1536)  # text-embedding-3-small has 1536 dims
        
        return results

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def normalize_answer(self, s: str) -> str:
        """
        Normalize the input string by lowering case, removing punctuation, articles, and extra whitespace.
        """

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace("_", " ")

        return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))

    def calculate_f1_score(self, prediction: str, reference: str) -> float:
        """
        Calculate the F1 score between a prediction and a reference string after normalization.
        """
        prediction = str(prediction)
        reference = str(reference)
        prediction_tokens = self.normalize_answer(prediction).split()
        reference_tokens = self.normalize_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_code_bleu(
        self, prediction: str, reference: str, lang: str = "python"
    ) -> float:
        """
        Compute the CodeBLEU score for a single prediction and reference code snippet.
        """
        results = []
        score = calc_codebleu([prediction], [reference], lang="python")
        return score["codebleu"]

    def calculate_bleu_score(self, prediction: str, reference: str) -> float:
        """
        Calculate the BLEU score between a prediction and a reference string after normalization.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        reference_tokens = self.normalize_answer(reference).split()
        bleu = sentence_bleu(reference_tokens, prediction_tokens)
        return bleu

    def compute_semantic_similarity(self, prediction: str, reference: str) -> float:
        """
        Compute the semantic similarity between a prediction and a reference using OpenAI embeddings and cosine similarity.
        """
        prediction = str(prediction)
        reference = str(reference)
        embedding1 = self.get_openai_embedding(
            prediction, model="text-embedding-3-small"
        )
        embedding2 = self.get_openai_embedding(
            reference, model="text-embedding-3-small"
        )
        similarity_score = cosine_similarity(
            embedding1.reshape(1, -1), embedding2.reshape(1, -1)
        )
        return similarity_score[0][0]

    def compute_semantic_similarity_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Compute semantic similarity for multiple prediction-reference pairs efficiently using batch processing.
        """
        # Collect all unique texts
        all_texts = list(set(str(p) for p in predictions) | set(str(r) for r in references))
        
        # Get embeddings for all texts in batch
        embeddings_dict = self.get_openai_embeddings_batch(all_texts, model="text-embedding-3-small")
        
        # Compute similarities
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_str = str(pred)
            ref_str = str(ref)
            
            pred_embedding = embeddings_dict.get(pred_str)
            ref_embedding = embeddings_dict.get(ref_str)
            
            if pred_embedding is not None and ref_embedding is not None:
                similarity_score = cosine_similarity(
                    pred_embedding.reshape(1, -1), ref_embedding.reshape(1, -1)
                )
                similarities.append(similarity_score[0][0])
            else:
                # Fallback to 0 if embedding failed
                similarities.append(0.0)
                
        return similarities

    def compute_word_similarity(self, prediction: str, reference: str) -> float:
        """
        Compute the word-level similarity between a prediction and a reference using TF-IDF and cosine similarity.
        """
        tfidf_pred = self.vectorizer.fit_transform([prediction, reference])
        sim = cosine_similarity(tfidf_pred[0:1], tfidf_pred[1:2])[0][0]
        return sim

    def calculate_accuracy_score(self, prediction: str, reference: str) -> float:
        """
        Calculate the accuracy score for a single prediction and reference (exact match).
        """
        prediction = str(prediction).lower().strip()
        reference = str(reference).lower().strip()
        return accuracy_score([prediction], [reference])
    
    def calculate_numerical_accuracy_score(self, prediction: str, reference: str) -> float:
        """
        Calculate the numerical accuracy score for a single prediction and reference (exact match).
        """
        prediction = str(self.extract_number(prediction))
        reference = str(self.extract_number(reference))
        if prediction is not None and reference is not None:
            return accuracy_score([prediction], [reference])
        else:
            return 0.0

    def calculate_r2_score(self, predictions, references) -> float:
        """
        Calculate the R2 score (regression metric) for a list of predictions and references.
        Normalize to [0, 1].
        """
        pred_nums = []
        ref_nums = []
        for pred, ref in zip(predictions, references):
            pred_num = self.extract_number(pred)
            ref_num = self.extract_number(ref)
            if pred_num is not None and ref_num is not None:
                pred_nums.append(pred_num)
                ref_nums.append(ref_num)
        if len(pred_nums) >= 2:
            r2 = r2_score(ref_nums, pred_nums)
            return max(0.0, min(1.0, r2))
        else:
            return 0.0

    def __call__(
        self,
        calculator_type: MetricType,
        predictions: pd.DataFrame,
        references: pd.DataFrame,
    ) -> Tuple[float, pd.DataFrame]:
        references["id"] = references["id"].astype(str)
        predictions["id"] = predictions["id"].astype(str)
        merged = pd.merge(references, predictions, on="id", how="left")
        
        # Handle prediction: keep integer, only fillna for empty values
        merged["prediction"] = merged["prediction"].fillna("")
        
        def safe_convert_to_str(x):
            if pd.isna(x) or x == "":
                return ""
            try:
                # If x is a number and an integer, convert to int before string
                if isinstance(x, (int, float)) and float(x).is_integer():
                    return str(int(float(x)))
                else:
                    return str(x)
            except:
                return str(x)
        
        merged["prediction"] = merged["prediction"].apply(safe_convert_to_str)
        merged["label"] = merged["label"].apply(safe_convert_to_str)
        scores = []

        if calculator_type == MetricType.R2:
            average_score = self.calculate_r2_score(
                merged["prediction"], merged["label"]
            )
            merged["score"] = [average_score] * len(merged)
        elif calculator_type == MetricType.SEMANTIC_SIMILARITY:
            # Use batch processing for semantic similarity for much better performance
            scores = self.compute_semantic_similarity_batch(
                merged["prediction"].tolist(), merged["label"].tolist()
            )
            merged["score"] = scores
            average_score = sum(scores) / len(scores) if scores else 0.0
        else:
            for _, row in merged.iterrows():
                if calculator_type == MetricType.CODE_BLEU:
                    score = self.compute_code_bleu(row["prediction"], row["label"])
                elif calculator_type == MetricType.BLEU:
                    score = self.calculate_bleu_score(row["prediction"], row["label"])
                elif calculator_type == MetricType.WORD_SIMILARITY:
                    score = self.compute_word_similarity(
                        row["prediction"], row["label"]
                    )
                elif calculator_type == MetricType.ACCURACY:
                    score = self.calculate_accuracy_score(
                        row["prediction"], row["label"]
                    )
                elif calculator_type == MetricType.F1:
                    score = self.calculate_f1_score(row["prediction"], row["label"])
                elif calculator_type == MetricType.NUMERICAL_ACCURACY:
                    score = self.calculate_numerical_accuracy_score(
                        row["prediction"], row["label"]
                    )
                else:
                    raise ValueError(f"Unknown calculator type: {calculator_type}")
                scores.append(score)
            merged["score"] = scores
            average_score = sum(scores) / len(scores) if scores else 0.0

        merged["calculation_type"] = calculator_type.value
        return average_score, merged


if __name__ == "__main__":
    calculator = Calculator()
    predictions = pd.read_csv("tasks/chain-level/code_generation/predictions.csv")
    references = pd.read_csv("tasks/chain-level/code_generation/labels.csv")
    print(calculator("code_bleu", predictions, references))
