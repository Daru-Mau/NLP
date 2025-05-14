# inference.py

import argparse
import torch
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForMultipleChoice


def load_baseline_model(model_path, vectorizer_path):
    """
    Load the baseline logistic regression model and TF-IDF vectorizer.

    Args:
        model_path: Path to the saved model
        vectorizer_path: Path to the saved vectorizer

    Returns:
        Loaded model and vectorizer
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def load_bert_model(model_dir):
    """
    Load the BERT model and tokenizer.

    Args:
        model_dir: Directory containing the saved model

    Returns:
        Loaded model and tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForMultipleChoice.from_pretrained(model_dir)

    return model, tokenizer


def predict_baseline(model, vectorizer, article, question, options):
    """
    Make a prediction using the baseline model.

    Args:
        model: Trained logistic regression model
        vectorizer: Trained TF-IDF vectorizer
        article: Text article
        question: Question about the article
        options: List of possible answers

    Returns:
        Predicted answer index and probabilities
    """
    texts = [f"{article} {question} {option}" for option in options]
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)[:, 1]  # Probability of positive class
    pred_idx = np.argmax(probs)

    return pred_idx, probs


def predict_bert(model, tokenizer, article, question, options, device='cpu'):
    """
    Make a prediction using the BERT model.

    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        article: Text article
        question: Question about the article
        options: List of possible answers
        device: 'cpu' or 'cuda'

    Returns:
        Predicted answer index and probabilities
    """
    inputs = [f"{article} [SEP] {question} [SEP] {option}" for option in options]

    encoding = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    model.to(device)
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)

    return pred_idx, probs


def format_answer(idx):
    """Convert numeric index to A, B, C, D format"""
    return chr(ord('A') + idx)


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions on RACE-style questions')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['baseline', 'bert'],
                        help='Type of model to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (directory for BERT, file for baseline)')
    parser.add_argument('--vectorizer_path', type=str,
                        help='Path to TF-IDF vectorizer (only needed for baseline)')
    parser.add_argument('--article_file', type=str, required=True,
                        help='File containing the article text')
    parser.add_argument('--question', type=str, required=True,
                        help='Question about the article')
    parser.add_argument('--options', type=str, nargs='+', required=True,
                        help='List of answer options')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    args = parser.parse_args()

    # Load article
    with open(args.article_file, 'r', encoding='utf-8') as f:
        article = f.read()

    # Determine device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # Make prediction based on model type
    if args.model_type == 'baseline':
        if args.vectorizer_path is None:
            raise ValueError(
                "Vectorizer path must be provided for baseline model")

        model, vectorizer = load_baseline_model(
            args.model_path, args.vectorizer_path)
        pred_idx, probs = predict_baseline(
            model, vectorizer, article, args.question, args.options)
    else:  # BERT
        model, tokenizer = load_bert_model(args.model_path)
        pred_idx, probs = predict_bert(
            model, tokenizer, article, args.question, args.options, device)

    # Display results
    print(f"Question: {args.question}\n")

    for i, (option, prob) in enumerate(zip(args.options, probs)):
        marker = "✓" if i == pred_idx else " "
        print(f"{marker} {format_answer(i)}: {option} ({prob:.4f})")

    print(f"\nPredicted answer: {format_answer(pred_idx)}")


if __name__ == "__main__":
    main()
