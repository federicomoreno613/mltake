import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score


def model_eval(model, X_train, X_test, y_train, y_test,X_val,Y_val, target_names=["Not Fraud", "Fraud"]):
    """
    Evaluate a classification model with metrics and visualizations.

    Args:
    model: The trained model to evaluate.
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing/validation features.
    y_train (Series): Training target.
    y_test (Series): Testing/validation target.
    target_names (list): List of target class names.
    """

    # Predictions on test set
    y_pred_test = model.predict(X_test)
    #print separator
    print("--------------------------------------------------")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_test)}")

    # Other metrics
    print(f"Precision: {precision_score(y_test, y_pred_test)}")
    print(f"Recall: {recall_score(y_test, y_pred_test)}")
    print(f"F1: {f1_score(y_test, y_pred_test)}")
    # Confusion Matrix on test set
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print("Test Confusion Matrix:")
    sns.heatmap(conf_matrix_test, annot=True, fmt='g', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on Test set')
    plt.show()

    # ROC AUC on test set
    print("--------------------------------------------------")
    # ROC AUC on validation set
    print(f"ROC AUC: {roc_auc_score(Y_val, model.predict(X_val))}")
    print(f"Precision: {precision_score(Y_val, model.predict(X_val))}")
    print(f"Recall: {recall_score(Y_val, model.predict(X_val))}")
    print(f"F1: {f1_score(Y_val, model.predict(X_val))}")
    #confusion matriz on validation set
    conf_matrix_test = confusion_matrix(Y_val, model.predict(X_val))
    print("Test Confusion Matrix:")
    sns.heatmap(conf_matrix_test, annot=True, fmt='g', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on validation set')
    plt.show()

    # SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)

# Example usage
# model_eval(best_xgb_model, X_train, X_test, y_train, y_test)


def calculate_metrics(y_true, y_pred,model_name,set):
    """
    Calculate classification metrics and return them as a DataFrame.

    Args:
    y_true (Series): True target values.
    y_pred (Series): Predicted target values.

    Returns:
    DataFrame: A DataFrame with the calculated metrics.
    """
    metrics = {
        'model_name':model_name,
        'set': set,
        'ROC AUC': roc_auc_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    return pd.DataFrame([metrics], index=['Model Metrics'])

def plot_confusion_matrices(y_test, y_pred_test, Y_val, y_pred_val, target_names):
    """
    Plot confusion matrices for test and validation sets side by side.

    Args:
    y_test (Series): True target values for the test set.
    y_pred_test (Series): Predicted target values for the test set.
    Y_val (Series): True target values for the validation set.
    y_pred_val (Series): Predicted target values for the validation set.
    target_names (list): List of target class names.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Test set confusion matrix
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(conf_matrix_test, ax=axes[0], annot=True, fmt='g', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix on Test set')

    # Validation set confusion matrix
    conf_matrix_val = confusion_matrix(Y_val, y_pred_val)
    sns.heatmap(conf_matrix_val, ax=axes[1], annot=True, fmt='g', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix on Validation set')

    plt.tight_layout()
    plt.show()
