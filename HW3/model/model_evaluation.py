import numpy as np

class BinaryEvaluation:
    def __init__(self, nn_model):
        """
        Initializes the Evaluation class with a trained model.

        Parameters:
        nn_pop (object): A trained model that has a .predict(X) method.
        """
        self.nn_model = nn_model

    def model_predict(self, X):
        """
        Predicts output using the trained model.

        Parameters:
        X (np.ndarray): Input features.

        Returns:
        np.ndarray: Predicted values.
        """
        return self.nn_model.predict(X)

    def model_test(self, X_test, Y_test):
        """
        Generates class predictions and evaluates them using the evaluate method.

        Parameters:
        X_test (np.ndarray): Test features.
        Y_test (np.ndarray): True class labels.

        Returns:
        tuple: (evaluation_metrics, confusion_matrix)
        """
        # print("X_test", self.X_test)
        # print("Y_test", self.Y_test)
        y_hat = self.model_predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        # y_hat_class = np.where(y_hat.cpu()<0.5, 0, 1)
        # print("y_hat", self.y_hat)
        # print("y_hat_class", self.y_hat_class)
        # print("Y_test", self.Y_test)
        return self.evaluate(y_hat_class, Y_test)

    def evaluate(self, y_hat_class, Y):  # Y = real value, Y_hat = expected
        """
        Evaluates prediction performance and computes various classification metrics.

        Parameters:
        y_hat_class (np.ndarray): Predicted class labels.
        Y (np.ndarray): True class labels.

        Returns:
        tuple: (metrics array, confusion matrix)
        """
        cm = np.zeros((2, 2), dtype=int)
        y_hat_class = y_hat_class.reshape(-1)

        # Populate confusion matrix
        for y_hat, y in zip(y_hat_class, Y):
            if y == 0:
                if y_hat == 0:
                    cm[0, 0] += 1  # T0: actual 0, predicted 0 (TP for class 0)
                else:
                    cm[0, 1] += 1  # F0: actual 0, predicted 1 (FN for class 0)
            elif y == 1:
                if y_hat == 1:
                    cm[1, 1] += 1  # T1: actual 1, predicted 1 (TN for class 1)
                else:
                    cm[1, 0] += 1  # F1: actual 1, predicted 0 (FP for class 1)

        T0 = cm[0, 0]
        F0 = cm[0, 1]
        F1 = cm[1, 0]
        T1 = cm[1, 1]

        # Metrics
        recall_0 = T0 / (T0 + F0) if (T0 + F0) != 0 else 0.0  # Sensitivity for class 0
        specificity_1 = T1 / (T1 + F1) if (T1 + F1) != 0 else 0.0  # Specificity for class 1

        precision_0 = T0 / (T0 + F1) if (T0 + F1) != 0 else 0.0  # Precision for class 0
        precision_1 = T1 / (T1 + F0) if (T1 + F0) != 0 else 0.0  # Precision for class 1

        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) != 0 else 0.0
        f1_1 = 2 * precision_1 * specificity_1 / (precision_1 + specificity_1) if (precision_1 + specificity_1) != 0 else 0.0

        weighted_accuracy = 0.5 * (recall_0 + specificity_1)
        auc_roc = 0.5 * (1 + recall_0 - (F1 / (F1 + T1) if (F1 + T1) != 0 else 0.0))
        jaccard = T0 / (T0 + F0 + F1) if (T0 + F0 + F1) != 0 else 0.0

        fdr = F1 / (F1 + T0) if (F1 + T0) != 0 else 0.0
        fnr = F0 / (F0 + T0) if (F0 + T0) != 0 else 0.0
        forate = F0 / (F0 + T1) if (F0 + T1) != 0 else 0.0
        fpr = F1 / (F1 + T1) if (F1 + T1) != 0 else 0.0

        precision_avg = (precision_0 + precision_1) / 2
        f1_avg = (f1_0 + f1_1) / 2

        return (
            np.array([
                weighted_accuracy,
                recall_0,
                specificity_1,
                precision_0,
                precision_1,
                precision_avg,
                f1_0,
                f1_1,
                f1_avg,
                auc_roc,
                fdr,
                fnr,
                forate,
                fpr,
                jaccard,
            ]),
            cm
        )
        # cm = np.zeros(
        #     (2, 2), dtype=int
        # )  # Initialize the confusion matrix as a 2x2 matrix of zeros
        # y_hat_class = y_hat_class.reshape(-1)  # Ensure y_hat_class is a 1D array
        # # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
        # for y_hat, y in zip(y_hat_class, Y):
        #     if y == 0:  # Actual class is 0 (positive)
        #         if y_hat == 0:
        #             cm[0, 0] += 1  # True Positive (TP)
        #         else:
        #             cm[0, 1] += 1  # False Negative (FN)
        #     elif y == 1:  # Actual class is 1 (negative)
        #         if y_hat == 1:
        #             cm[1, 1] += 1  # True Negative (TN)
        #         else:
        #             cm[1, 0] += 1  # False Positive (FP)
        # # print("cm",cm)
        # # print("cm.ravel()",cm.ravel())
        # # tn, fp, fn, tp = cm.ravel()
        # tp, fn, fp, tn = cm.ravel()
        # assert (tp + tn + fp + fn) != 0.0
        # # a = (tp + tn) / (tp + tn + fp + fn)                                       ## Removed - Accuracy
        # wa0 = (
        #     (tn / (2 * (tn + fp))) if (tn + fp) != 0.0 else 0.0
        # )  ## Local Var - Weighted_Accuracy_class0
        # wa1 = (
        #     (tp / (2 * (tp + fn))) if (tp + fn) != 0.0 else 0.0
        # )  ## Local Var - Weighted_Accuracy_class1
        # wa = wa0 + wa1  ## Weighted Accuracy

        # r = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0  ## Sensitivity/Recall

        # p0 = (
        #     tn / (tn + fn) if (tn + fn) != 0.0 else 0.0
        # )  ## Precision_class0 # negative predictive value
        # p1 = (
        #     tp / (tp + fp) if (tp + fp) != 0.0 else 0.0
        # )  ## Precision_class1 # positive predictive value

        # s = tn / (tn + fp) if (tn + fp) != 0.0 else 0.0  ## Specificity

        # fscore0 = 2 * p0 * r / (p0 + r) if p0 + r != 0.0 else 0.0  ## F1_class0
        # fscore1 = 2 * p1 * s / (p1 + s) if p1 + s != 0.0 else 0.0  ## F1_class1
        # # d = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) != 0.0 else 0.0     ## Removed - Accuracy
        # j = tn / (tn + fp + fn) if (tn + fp + fn) != 0.0 else 0.0  ## Jaccard
        # tpr = (
        #     tp / (fn + tp) if (fn + tp) != 0 else 0.0
        # )  ## True Positive Rate - Local Var
        # fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0  ## False Positive Rate
        # tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0  ## True Negative Rate
        # fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0  ## False Negative Rate
        # fdr = fp / (tp + fp) if (tp + fp) != 0 else 0.0  ## False Discovery Rate
        # fo_rate = fn / (fn + tn) if (fn + tn) != 0 else 0.0  ## False Omission Rate
        
        # # Formula for AUC_Roc score without using function - https://stackoverflow.com/questions/50848163/manually-calculate-auc
        # auc_roc = (1 / 2) - (fpr / 2) + (tpr / 2)  ## auc_roc score
        # pavg = (p0 + p1) / 2.0  ## Precision_avg
        # f1avg = (fscore0 + fscore1) / 2.0  ## F1_avg
        
        # return (
        #     np.array(
        #         [
        #             # tp,
        #             # fn,
        #             # fp,
        #             # tn,
        #             wa,
        #             r,
        #             s,
        #             p0,
        #             p1,
        #             pavg,
        #             fscore0,
        #             fscore1,
        #             f1avg,
        #             auc_roc,
        #             fpr,
        #             fdr,
        #             fnr,
        #             fo_rate,
        #             j,
        #         ]
        #     ),
        #     cm,
        # )