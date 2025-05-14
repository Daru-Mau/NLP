# RACE Dataset Analysis - NLP Project

![GitHub](https://img.shields.io/github/license/yourusername/race-nlp-analysis)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/race-nlp-analysis)

This repository contains analysis of the Reading Comprehension from Examinations (RACE) dataset, a large-scale reading comprehension dataset collected from English examinations in China for middle and high school students.

## Quick Summary

The RACE (Reading Comprehension from Examinations) dataset is one of the largest in the reading comprehension domain, containing:

- 28,000+ passages and nearly 100,000 questions
- Middle and high school English exam materials
- Multiple-choice format (4 options per question)

This project explores the dataset through:

1. Comprehensive data analysis and visualization
2. Implementation of multiple models (baseline and BERT-based)
3. Comparison of model performance
4. Interactive search capabilities
5. Ready-to-use helper scripts for dataset handling

## Overview

The analysis includes:

- Dataset exploration and structure analysis
- Vocabulary analysis and visualization
- Document clustering
- Text search engine implementation
- Word embedding analysis with Word2Vec
- Multiple machine learning models for the reading comprehension task:
  - Baseline logistic regression model
  - BERT-based transformer model
  - Large Language Model approaches

![RACE Dataset](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/race_layout.png)

## Repository Structure

### Main Files

- `RACE_Analysis.ipynb`: Main Jupyter notebook containing all analysis and code
- `NLP_Project.ipynb`: Preliminary analysis notebook
- `requirements.txt`: Python package dependencies
- `.gitignore`: Standard file to exclude unnecessary files from Git
- `LICENSE`: MIT license file

### Helper Scripts

- `data_utils.py`: Utility functions for loading and processing the RACE dataset
- `train_models.py`: Script for training baseline and BERT models on the RACE dataset
- `inference.py`: Script for making predictions with trained models
- `STRUCTURE.md`: Detailed description of the project structure
- `CONTRIBUTING.md`: Guidelines for contributing to the project

## Setup and Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/race-nlp-analysis.git
cd race-nlp-analysis
```

2. Create a virtual environment (recommended):

```bash
python -m venv nlp_venv
# On Windows
nlp_venv\Scripts\activate
# On macOS/Linux
source nlp_venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook

Open the Jupyter notebook in your preferred environment:

```bash
jupyter notebook RACE_Analysis.ipynb
```

The notebook will automatically download the RACE dataset using the Hugging Face datasets library.

### Helper Scripts

You can also use the provided helper scripts to work with the dataset and models:

#### Loading the Dataset

```python
from data_utils import load_race_dataset, convert_to_dataframe

# Load the dataset
dataset = load_race_dataset()

# Convert to DataFrame
train_df = convert_to_dataframe(dataset['train'])
val_df = convert_to_dataframe(dataset['validation'])
test_df = convert_to_dataframe(dataset['test'])
```

#### Training Models

To train a baseline model (TF-IDF + Logistic Regression):

```bash
python train_models.py --model_type baseline --output_dir models
```

To train a BERT model:

```bash
python train_models.py --model_type bert --output_dir models --epochs 3
```

For faster testing with a smaller sample:

```bash
python train_models.py --model_type both --sample_size 1000
```

#### Making Predictions

Using the baseline model:

```bash
python inference.py --model_type baseline \
  --model_path models/baseline_model.pkl \
  --vectorizer_path models/tfidf_vectorizer.pkl \
  --article_file example_article.txt \
  --question "What is the main idea of the passage?" \
  --options "Option A" "Option B" "Option C" "Option D"
```

Using the BERT model:

```bash
python inference.py --model_type bert \
  --model_path models/bert \
  --article_file example_article.txt \
  --question "What is the main idea of the passage?" \
  --options "Option A" "Option B" "Option C" "Option D" \
  --cuda
```

## Dataset

The RACE dataset contains:

- Over 28,000 passages and nearly 100,000 questions
- English exam materials for middle and high school students in China
- Multiple-choice format with four options per question

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Results

The analysis explores:

- Text complexity and patterns in middle vs. high school exam questions
- Topic distribution across passages using clustering techniques
- Effectiveness of different embedding approaches (Word2Vec, BERT)
- Model performance comparison on the reading comprehension task
- Search engine efficiency using different indexing methods

Detailed findings and visualizations are available in the notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a Pull Request

## Citation

If you use this analysis in your work, please cite both our repository and the original RACE dataset:

```bibtex
@inproceedings{lai-etal-2017-race,
    title = "{RACE}: Large-scale {R}eading Comprehension Dataset From Examinations",
    author = "Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2017",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D17-1082",
    pages = "785--794",
}
```

## Acknowledgments

- [RACE dataset paper](https://www.aclweb.org/anthology/D17-1082/)
- [Hugging Face RACE dataset](https://huggingface.co/datasets/race)
- [Transformers library](https://github.com/huggingface/transformers)
