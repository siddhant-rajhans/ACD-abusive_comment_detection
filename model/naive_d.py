import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a sample dataset
comments = [
    {'comment_text': 'This is a great product!', 'is_abusive': 0},
    {'comment_text': 'I love this product!', 'is_abusive': 0},
    {'comment_text': 'This product is terrible.', 'is_abusive': 1},
    {'comment_text': 'I hate this product!', 'is_abusive': 1}
]
df = pd.DataFrame(comments)

# Save the sample dataset to a CSV file
df.to_csv('comments.csv', index=False)

# Load the data into a pandas dataframe
df = pd.read_csv('comments.csv')

# Split the data into features (X) and target (y)
X = df['comment_text']
y = df['is_abusive']

# Convert the text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)
