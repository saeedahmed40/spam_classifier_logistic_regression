import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read email data
def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df = pd.DataFrame.from_records(ham)
df = df.append(pd.DataFrame.from_records(spam))

# Preprocessing function
def preprocessor(e):
    e = re.sub(r'<[^>]*>', '', e)  # Remove HTML tags
    e = re.sub(r'[^a-zA-Z\s]', '', e)  # Remove non-alphabetic characters
    e = e.lower()  # Convert to lowercase
    return e

# Vectorizer
vectorizer = CountVectorizer(stop_words='english', preprocessor=preprocessor)

# Split data
X = df['content']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data into vectors
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Feature importance
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive_coefficients = coefficients.argsort()[-10:][::-1]
top_negative_coefficients = coefficients.argsort()[:10]

print("Top 10 spam-indicating words:")
for idx in top_positive_coefficients:
    print(f'{feature_names[idx]}: {coefficients[idx]}')

print("\nTop 10 ham-indicating words:")
for idx in top_negative_coefficients:
    print(f'{feature_names[idx]}: {coefficients[idx]}')