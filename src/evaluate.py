import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test):

    """
    Evaluate the model using classification report and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Classification Report
    print("\n--- Classification Report ---\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_

    # Plotting the Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()



# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# def evaluate_model(model, X_test, y_test):
#     """Evaluate model with metrics & visualizations"""

#     # Predict
#     y_pred = model.predict(X_test)

#     # 1. Classification Report
#     print(" Classification Report:\n")
#     print(classification_report(y_test, y_pred))

#     # 2. Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)

#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(' Confusion Matrix')
#     plt.tight_layout()
#     plt.show()

#     # 3. Print Accuracy, Precision, Recall, F1
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_label='Yes')
#     recall = recall_score(y_test, y_pred, pos_label='Yes')
#     f1 = f1_score(y_test, y_pred, pos_label='Yes')

#     print(f" Accuracy : {accuracy:.2f}")
#     print(f" Precision: {precision:.2f}")
#     print(f" Recall   : {recall:.2f}")
#     print(f" F1 Score : {f1:.2f}")
