# BERT-Text-Classification-for-Sentiment-Analysis

*OVERVIEW*

This project implements a BERT-based text classification model for sentiment analysis using the SMILE Twitter dataset. The model classifies tweets into six emotion categories: happy, not-relevant, angry, disgust, sad, and surprise. The dataset is highly imbalanced, so techniques like data augmentation (using nlpaug) and class-weighted loss are employed to improve performance on minority classes. The model is built using PyTorch and the Hugging Face Transformers library, with evaluation metrics including macro F1 score, precision, recall, and per-class accuracy.

*FEATURES*





Data Preprocessing: Cleans tweets by removing URLs and special characters.



Data Augmentation: Uses synonym-based augmentation (nlpaug) to address class imbalance for disgust, sad, and surprise.



Model: Fine-tunes bert-base-uncased for multiclass classification.



Training: Includes early stopping, class-weighted loss, and learning rate scheduling.



Evaluation: Reports per-class metrics (accuracy, precision, recall, F1) and visualizes confusion matrices using seaborn.



Reproducibility: Sets random seeds for consistent results.


*DATASET*

The dataset (smile-annotations-final.csv) contains tweets labeled with emotions. After preprocessing, the class distribution is:





happy: 1137 samples



not-relevant: 214 samples



angry: 57 samples



surprise: 35 samples



sad: 32 samples



disgust: 6 samples

Data augmentation increases the number of samples for disgust (~50), sad (~100), and surprise (~100) in the training set.
