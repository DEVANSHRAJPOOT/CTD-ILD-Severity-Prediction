from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import numpy as np


def evaluate_models(models, X_train, X_test, y_train, y_test, transformer):
    """
    Train and evaluate models with safe AUROC handling
    for binary and multiclass cases.
    """

    for name, model in models.items():
        pipe = Pipeline([
            ("transform", transformer),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print("=" * 60)
        print(f"MODEL: {name}")
        print(classification_report(y_test, y_pred))

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)
            n_classes = len(np.unique(y_test))

            try:
                if n_classes == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                    print("AUROC (binary):", round(auc, 3))
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                    print("AUROC (multiclass OvR):", round(auc, 3))
            except ValueError:
                print("AUROC: not defined for this split")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
