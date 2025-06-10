import re
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def process_review(reviews):
    cleaned = [preprocess_text(r) for r in reviews]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned)
    vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]))
    
    print("\n📘 Vocabulary:")
    for word, idx in vocab.items():
        print(f"{word}: {idx}")
    
    print("\n🧾 Bag of Words Matrix:")
    for row in X.toarray():
        print(row)

# Example
if __name__ == "__main__":
    reviews = [
        "AI is amazing!",
        "I love AI and Machine Learning.",
        "AI is the future."
    ]
    process_review(reviews)
