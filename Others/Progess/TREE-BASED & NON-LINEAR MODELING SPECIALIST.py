import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_advanced_models():
    print("Loading extracted features and labels...")
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(probability=True, kernel='rbf', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    # Hybrid Model Construction
    hybrid_estimators = [
        ('lr', LogisticRegression(max_iter=2000)),
        ('rf', models["Random Forest"]),
        ('svm', models["SVM (RBF Kernel)"]),
        ('xgb', models["XGBoost"])
    ]
    models["Hybrid Model"] = VotingClassifier(estimators=hybrid_estimators, voting='soft')

    results = []
    print("\n--- Training Non-Linear and Ensemble Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Test Accuracy': acc})
        print(f"✅ {name} finished. Test Accuracy: {acc * 100:.2f}%")

    results_df = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    
    # Export Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test Accuracy', y='Model', data=results_df)
    plt.title('Non-Linear & Ensemble Model Comparison')
    plt.axvline(0.90, color='red', linestyle='--', label='90% Target')
    plt.xlim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("advanced_models_accuracy.png")
    print("\nSaved performance plot as 'advanced_models_accuracy.png'")

if __name__ == "__main__":
    run_advanced_models()