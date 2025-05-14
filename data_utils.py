# data_utils.py

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple, Union, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources


def download_nltk_resources():
    """Download required NLTK resources if they don't exist."""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)


def load_race_dataset(cache_dir: Optional[str] = None) -> Dict:
    """
    Load the RACE dataset from Hugging Face.

    Args:
        cache_dir: Optional directory where to cache the dataset

    Returns:
        Dictionary containing train, validation, and test splits
    """
    print("Loading RACE dataset from Hugging Face...")

    dataset = load_dataset("race", "all", cache_dir=cache_dir)

    print(f"Dataset loaded with {len(dataset['train'])} training examples, "
          f"{len(dataset['validation'])} validation examples, and "
          f"{len(dataset['test'])} test examples.")

    return dataset


def convert_to_dataframe(dataset_split) -> pd.DataFrame:
    """
    Convert a RACE dataset split to a pandas DataFrame for easier analysis.

    Args:
        dataset_split: A split from the RACE dataset (e.g., dataset['train'])

    Returns:
        DataFrame containing the dataset
    """
    data = []

    for item in dataset_split:
        article_id = item['article_id']
        example_id = item['example_id']
        answer = item['answer']
        options = item['options']
        question = item['question']
        article = item['article']

        # Create one row per question
        data.append({
            'article_id': article_id,
            'example_id': example_id,
            'article': article,
            'question': question,
            'options': options,
            'answer': answer,
            # Convert A,B,C,D to actual answer text
            'answer_text': options[ord(answer) - ord('A')],
            # True for high school, False for middle school
            'high': item['high']
        })

    return pd.DataFrame(data)


def preprocess_text(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Preprocess text by tokenizing and optionally removing stopwords.

    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords

    Returns:
        List of preprocessed tokens
    """
    download_nltk_resources()

    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

    return tokens


def get_difficulty_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the distribution of questions by difficulty level.

    Args:
        df: DataFrame containing RACE dataset

    Returns:
        DataFrame with question counts by difficulty level
    """
    difficulty_counts = df.groupby('high').size().reset_index()
    difficulty_counts.columns = ['is_high_school', 'count']
    difficulty_counts['level'] = difficulty_counts['is_high_school'].apply(
        lambda x: 'High School' if x else 'Middle School')

    return difficulty_counts[['level', 'count']]


if __name__ == "__main__":
    # Example usage
    download_nltk_resources()
    dataset = load_race_dataset()
    train_df = convert_to_dataframe(dataset['train'])
    print(f"Converted training set to DataFrame with shape: {train_df.shape}")
    print("\nDifficulty distribution:")
    print(get_difficulty_distribution(train_df))
