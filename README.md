# Fake News Detection Project

## Overview
This project aims to identify and classify fake news articles using machine learning techniques. By leveraging Natural Language Processing (NLP) and various machine learning algorithms, the project analyzes text data to distinguish between real and fake news.

## Dataset
The dataset used for this project was sourced from Kaggle. You can find the dataset [here](https://www.kaggle.com/competitions/fake-news/data). It contains various text data labeled as either fake or real, which serves as the foundation for training and evaluating the model.

## Libraries Used
The following libraries were utilized in this project:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `re`: For regular expression operations.
- `nltk`: For natural language processing tasks, including stopword removal and stemming.
- `sklearn`: For machine learning algorithms, model evaluation, and text vectorization.

## Implementation Steps

1. **Data Preprocessing**:
   - Loaded the dataset and performed initial exploration.
   - Cleaned the text data by replacing null values with empty strings, removing special characters, stopwords, and applying stemming.

2. **Feature Extraction**:
   - Utilized `TfidfVectorizer` to convert the text data into numerical format suitable for machine learning models.

3. **Model Training**:
   - Split the dataset into training and testing sets.
   - Implemented `Logistic Regression` as the classification algorithm.

4. **Model Evaluation**:
   - Evaluated the model's performance using accuracy score.

## Results
The model achieved an accuracy of 98.7% on the training data and 97.5% on the test data.

## Conclusion
This project demonstrates the effectiveness of machine learning techniques in combating the spread of fake news. Future work may include expanding the dataset, implementing more complex models, and enhancing the preprocessing steps.


## Acknowledgments
- Kaggle for providing the dataset.
- The authors of the libraries and resources used in this project.
