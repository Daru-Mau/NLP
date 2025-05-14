# Project Structure

This document describes the organization of this repository.

## Main Files

- `RACE_Analysis.ipynb`: Primary analysis notebook containing the complete workflow from data loading to model evaluation
- `NLP_Project.ipynb`: Preliminary analysis notebook with initial data exploration
- `README.md`: Project overview, setup instructions, and other important information
- `requirements.txt`: List of Python package dependencies
- `LICENSE`: MIT license file
- `STRUCTURE.md`: This file explaining the project organization

## Data

The RACE dataset is not included in this repository due to size constraints. The analysis notebooks automatically download the dataset using the Hugging Face datasets library.

### Dataset Format

The RACE dataset includes:

- **Passages**: Text passages from English exams
- **Questions**: Multiple-choice questions about the passages
- **Options**: Four possible answers for each question
- **Answers**: Correct answers to the questions
- **Metadata**: Additional information such as article ID, exam level (middle/high school)

## Analysis Flow

1. **Data Loading and Preprocessing**

   - Load the RACE dataset
   - Text normalization and cleaning
   - Feature extraction

2. **Exploratory Data Analysis**

   - Statistical analysis of text properties
   - Visualization of key metrics
   - Comparison of middle vs. high school texts

3. **Natural Language Processing Tasks**

   - Vocabulary analysis
   - Document clustering
   - Word embedding with Word2Vec
   - BERT-based analysis

4. **Model Development**

   - Baseline models
   - BERT-based transformer models
   - Performance evaluation

5. **Text Search Engine**
   - Implementation of a search engine for the dataset
   - Evaluation of search performance

## Generated Files

When running the notebooks, the following files/directories may be created:

- `.ipynb_checkpoints/`: Jupyter notebook checkpoints
- Various model weights files (if models are saved)
- Visualization outputs
