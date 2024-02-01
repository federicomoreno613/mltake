import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from takehome.output_calibrator import compute_binary_score


def evaluate_model(model, df_val, df_test, model_cols, labels_val, labels_test, model_type, name, estimator_metrics):
    # Determine the prediction method based on the model type
    if model_type == "logistic_regression":
        pred_val = model.predict(df_val[model_cols])
        pred_test = model.predict(df_test[model_cols])
    elif model_type == "xgboost":
        pred_val = model.predict_proba(df_val[model_cols].values)[:, 1]
        pred_test = model.predict_proba(df_test[model_cols].values)[:, 1]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Compute and store validation metrics
    metrics_dict_val = compute_binary_score(labels_val, pred_val)
    metrics_dict_val["name"] = name
    metrics_dict_val["model"] = model_type
    metrics_dict_val["stage"] = "val"
    estimator_metrics.append(metrics_dict_val)

    # Compute and store test metrics
    metrics_dict_test = compute_binary_score(labels_test, pred_test)
    metrics_dict_test["name"] = name
    metrics_dict_test["stage"] = "test"
    metrics_dict_val["model"] = model_type
    estimator_metrics.append(metrics_dict_test)

    return estimator_metrics


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



