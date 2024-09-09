# Spam Classifier using Logistic Regression

This project is a machine learning-based spam classifier built using Python, Scikit-learn, and the Enron email dataset. The classifier uses **Logistic Regression** to distinguish between spam and non-spam (ham) emails by converting the text into a numerical format using the **CountVectorizer** and training a model to recognize patterns commonly found in spam emails.

## Features

* **Preprocessing**: Emails are cleaned by removing HTML tags, special characters, and converting the text to lowercase.
* **Vectorization**: The email content is transformed into a numerical representation using the Bag-of-Words approach (`CountVectorizer`), which converts text into a vector of word counts.
* **Model Training**: Logistic Regression is used to classify emails as either spam or ham.
* **Model Evaluation**: The model is evaluated using metrics such as accuracy, confusion matrix, and classification report to determine its performance.
* **Feature Importance**: The most important words contributing to the classification decision are extracted, helping to identify key terms that indicate spam.

## Dataset

The project utilizes the **Enron email dataset**, which contains real email data labeled as spam and ham. This dataset is publicly available for research purposes.

## Installation

To run this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/spam-classifier.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Download the Enron dataset and place it in the `data/` directory.
4. Run the Jupyter notebook or script to train the model and evaluate its performance.

## Usage

* The **Jupyter notebook** walks you through the entire process of training the model, testing its performance, and interpreting results.
* The **trained model** can be used to classify new emails by transforming them into the same feature space using the `CountVectorizer`.

## Technologies Used

* Python
* Scikit-learn
* Pandas
* Numpy
* Jupyter Notebook

## Future Improvements

* Add additional text preprocessing steps like stemming or lemmatization.
* Experiment with different machine learning algorithms like Naive Bayes or SVM.
* Expand the dataset or integrate other spam datasets to improve generalization.
