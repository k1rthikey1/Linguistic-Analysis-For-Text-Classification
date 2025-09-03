import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Text Preprocessor
class BasicTextPreprocessor:
    def analyze(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return ' '.join(text.split())

# Default dataset
texts = [
    "Ronaldo scored a hat trick in the match",              # sports
    "The game of cricket was exciting",                     # sports
    "The stock market crashed due to inflation",            # finance
    "Investors are nervous about rising interest rates",    # finance
    "Apple released a new iPhone model",                    # technology
    "Android 12 brings new features",                       # technology
    "The movie had amazing visual effects",                 # entertainment
    "The romantic comedy was a big hit",                    # entertainment
    "SpaceX successfully launched its rocket",              # science
    "NASA discovered water on the moon",                    # science
]

labels = [
    "sports",
    "sports",
    "finance",
    "finance",
    "technology",
    "technology",
    "entertainment",
    "entertainment",
    "science",
    "science"
]

# Preprocess texts
preprocessor = BasicTextPreprocessor()
processed_texts = [preprocessor.analyze(t) for t in texts]

# Split data
X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.3, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Show classification report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(set(labels)))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(set(labels)))
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# User prediction loop
while True:
    user_input = input("\nEnter a sentence to classify (or type 'exit' to quit): ")
    if user_input.strip().lower() == 'exit':
        break
    cleaned_input = preprocessor.analyze(user_input)
    prediction = pipeline.predict([cleaned_input])[0]
    print(f"🔍 Predicted Category: {prediction}")