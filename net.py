import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open('./dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text


topics = []
labels = []
solutions = []

for item in data:
    topics.append(preprocess(item['Topic']))  # Применяем предобработку
    labels.append(item['label'])
    solutions.append(item.get('Solution', None))

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(topics, labels, range(len(topics)),
                                                                                 test_size=0.2, random_state=42)

# Преобразуем текстовые данные в векторы с помощью TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Обучаем модель классификации на метки (labels)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Проверка точности классификатора на тестовой выборке
y_pred = classifier.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Создаем словарь решений для быстрого доступа
solution_dict = {}
for label, solution in zip(labels, solutions):
    if solution is not None:
        solution_dict[label] = solution

# Шаблоны ответов для отсутствующих решений
response_templates = {
    'Support': "Try restarting your computer and checking for driver updates.",
    'Hardware': "Check the connections and make sure that all components are working properly.",
    'Software': "Make sure that you have the latest versions of the software installed.",
}


# Функция для получения решения по запросу
def get_solution(query):
    query_tfidf = vectorizer.transform([preprocess(query)])
    predicted_label = classifier.predict(query_tfidf)[0]

    label_probabilities = classifier.predict_proba(query_tfidf)[0]

    # Вывод вероятностей для всех label
    label_confidences = sorted(zip(classifier.classes_, label_probabilities * 100), key=lambda x: x[1], reverse=True)
    print(f"Predicted label: {predicted_label}")
    print("Probabilities for all labels:")
    for label, conf in label_confidences:
        print(f"{label}: {conf:.2f}%")

    # Ищем готовое решение для предсказанного label
    global solution
    solution = solution_dict.get(predicted_label)

    label_indices = [i for i, label in enumerate(y_train) if label == predicted_label]
    label_topics_tfidf = X_train_tfidf[label_indices]
    similarities = cosine_similarity(query_tfidf, label_topics_tfidf).flatten()
    similar_indices = np.array(label_indices)[similarities.argsort()[-5:][::-1]]

    # Получаем похожие темы и соответствующие решения
    similar_topics = [(X_train[i], solutions[train_indices[i]]) for i in similar_indices]

    problems_and_similar_solutions = []

    for i, (topic, similar_solution) in enumerate(similar_topics, start=1):
        if similar_solution:
            problems_and_similar_solutions.append([topic, similar_solution])
        else:
            problems_and_similar_solutions.append([topic, "There is no similar solution"])


    if solution is None:
        solution = response_templates.get(predicted_label,
                                          "We don't have an exact solution, but we recommend contacting a specialist.")


    return [solution, predicted_label, label_confidences, problems_and_similar_solutions]
