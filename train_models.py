# train_models.py

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizer,
    BertForMultipleChoice,
    TrainingArguments,
    Trainer
)
import json
import os
from data_utils import load_race_dataset, convert_to_dataframe


def train_baseline_model(train_df, val_df=None, test_df=None):
    """
    Train a baseline TF-IDF + Logistic Regression model.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)

    Returns:
        Trained model and accuracy metrics
    """
    print("Training baseline TF-IDF + Logistic Regression model...")

    # Prepare training data
    # We'll concatenate article + question + each option and make it a binary classification task
    X_train = []
    y_train = []

    for _, row in train_df.iterrows():
        article = row['article']
        question = row['question']
        options = row['options']
        answer = ord(row['answer']) - ord('A')  # Convert A,B,C,D to 0,1,2,3

        for i, option in enumerate(options):
            text = f"{article} {question} {option}"
            X_train.append(text)
            y_train.append(1 if i == answer else 0)

    # Train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    results = {'train_accuracy': model.score(X_train_tfidf, y_train)}

    # Evaluate on validation set if provided
    if val_df is not None:
        X_val = []
        y_val = []
        option_groups = []  # To group options for the same question
        current_group = []

        for _, row in val_df.iterrows():
            article = row['article']
            question = row['question']
            options = row['options']
            # Convert A,B,C,D to 0,1,2,3
            answer = ord(row['answer']) - ord('A')

            group_idx = len(option_groups)
            current_group = []

            for i, option in enumerate(options):
                text = f"{article} {question} {option}"
                X_val.append(text)
                y_val.append(1 if i == answer else 0)
                current_group.append(len(X_val) - 1)  # Index in X_val

            option_groups.append(current_group)

        X_val_tfidf = vectorizer.transform(X_val)
        # Probability of positive class
        val_probs = model.predict_proba(X_val_tfidf)[:, 1]

        # Compute accuracy at the question level (select option with highest probability)
        correct = 0
        total = len(option_groups)

        for group in option_groups:
            group_probs = [val_probs[idx] for idx in group]
            pred_option = np.argmax(group_probs)
            true_option = np.argmax([y_val[idx] for idx in group])
            if pred_option == true_option:
                correct += 1

        val_accuracy = correct / total
        results['validation_accuracy'] = val_accuracy
        print(f"Validation accuracy: {val_accuracy:.4f}")

    # Similar evaluation for test set if provided
    if test_df is not None:
        # (Implementation similar to validation evaluation)
        pass

    return {
        'model': model,
        'vectorizer': vectorizer,
        'results': results
    }


def train_bert_model(train_df, val_df=None, output_dir="bert_model", epochs=3):
    """
    Train a BERT model for multiple choice.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        output_dir: Directory to save model
        epochs: Number of training epochs

    Returns:
        Trained model and evaluation metrics
    """
    print("Training BERT model for multiple choice...")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMultipleChoice.from_pretrained("bert-base-uncased")

    # Function to encode examples
    def encode_examples(df, tokenizer, max_length=512):
        encodings = []
        labels = []

        for _, row in df.iterrows():
            article = row['article']
            question = row['question']
            options = row['options']
            # Convert A,B,C,D to 0,1,2,3
            answer = ord(row['answer']) - ord('A')

            # Prepare input by combining article, question, and each option
            inputs = []
            for option in options:
                inputs.append(f"{article} [SEP] {question} [SEP] {option}")

            # Tokenize
            encoding = tokenizer(
                inputs,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            encodings.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'labels': torch.tensor(answer)
            })

            labels.append(answer)

        return encodings, labels

    # Encode datasets
    train_encodings, train_labels = encode_examples(train_df, tokenizer)

    # Create PyTorch datasets
    class MultipleChoiceDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings[idx].items()}

        def __len__(self):
            return len(self.encodings)

    train_dataset = MultipleChoiceDataset(train_encodings)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_df is not None else "no",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

    # Return evaluation metrics
    return {
        'model': model,
        'tokenizer': tokenizer,
        'results': trainer.state.log_history
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models on RACE dataset')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'bert', 'both'],
                        help='Type of model to train')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (for BERT model)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size to use (for faster testing)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = load_race_dataset()

    # Convert to DataFrames
    train_df = convert_to_dataframe(dataset['train'])
    val_df = convert_to_dataframe(dataset['validation'])

    # Use a sample if specified (for faster testing)
    if args.sample_size:
        train_df = train_df.sample(min(args.sample_size, len(train_df)))
        val_df = val_df.sample(min(args.sample_size // 5, len(val_df)))

    # Train models based on specified type
    if args.model_type in ['baseline', 'both']:
        baseline_result = train_baseline_model(train_df, val_df)

        # Save results
        with open(f"{args.output_dir}/baseline_results.json", 'w') as f:
            json.dump(baseline_result['results'], f)

        print(
            f"Baseline model results saved to {args.output_dir}/baseline_results.json")

    if args.model_type in ['bert', 'both']:
        bert_output_dir = f"{args.output_dir}/bert"
        bert_result = train_bert_model(
            train_df, val_df, bert_output_dir, args.epochs)

        # Save results
        with open(f"{args.output_dir}/bert_results.json", 'w') as f:
            json.dump(bert_result['results'], f)

        print(
            f"BERT model results saved to {args.output_dir}/bert_results.json")
