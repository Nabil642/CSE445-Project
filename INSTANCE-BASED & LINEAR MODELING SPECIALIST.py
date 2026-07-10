import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_linear_baselines():
    print("Loading extracted features and labels...")
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Ridge Classifier": RidgeClassifier(),
        "KNN (5)": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    print("\n--- Training Linear and Instance-Based Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Test Accuracy': acc})

        print(f"\n✅ {name} finished. Test Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    print("\nLinear Baselines Summary:")
    print(results_df)

if __name__ == "__main__":
    run_linear_baselines()
