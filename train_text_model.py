import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Read dataset
data = pd.read_csv("datasets/merged_email_sms_spam_dataset.csv")

# 2. Keep only needed columns
data = data[['text', 'label']]

# 3. Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

data['text'] = data['text'].apply(clean_text)

# 4. Split into X and y
X = data['text']
y = data['label']

# 5. Convert text to numbers (vectorizer)
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 7. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Check accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "models/sms_email_model.pkl")
joblib.dump(vectorizer, "models/sms_email_vectorizer.pkl")

print("✅ SMS + Email model saved successfully!")
