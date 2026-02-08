import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load merged URL dataset
data = pd.read_csv("datasets/merged_url_dataset.csv")

# 2. Keep only required columns
data = data[['url', 'type']]

# 3. Clean URLs
def clean_url(url):
    url = str(url).lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www.", "", url)
    return url

data['url'] = data['url'].apply(clean_url)

X = data['url']
y = data['type']

# 4. Convert URLs to numbers (character-based TF-IDF)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
X_vec = vectorizer.fit_transform(X)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save model and vectorizer
joblib.dump(model, "models/link_model.pkl")
joblib.dump(vectorizer, "models/link_vectorizer.pkl")

print("✅ Raw Link + QR model saved successfully!")
