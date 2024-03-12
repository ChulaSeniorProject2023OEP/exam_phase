import pandas as pd
import cv2

df = pd.read_csv('dataset.csv')

def preprocess_image(extracted_frames, target_size=(224, 224)):
    # Load image
    img = cv2.imread(extracted_frames)
    # Resize image
    img = cv2.resize(img, target_size)
    # Normalize pixel values (optional)
    img = img / 255.0
    return img

X = []  # Feature list
y = df['label'].values  # Labels

for img_path in df['frame_path']:
    img = preprocess_image(img_path)
    X.append(img)

# Convert to numpy arrays
import numpy as np
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Initialize different models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

# Flatten the images for RandomForest
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

from sklearn.metrics import accuracy_score, classification_report

# Dictionary to hold accuracy scores
accuracy_scores = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    print(f"{name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
