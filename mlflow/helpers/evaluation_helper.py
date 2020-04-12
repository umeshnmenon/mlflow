import numpy as np
import pandas as pd
import logging
from autologging import logged
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,average_precision_score, recall_score, f1_score, roc_curve ,auc, matthews_corrcoef, roc_auc_score, classification_report

class EvaluationHelper:
    """
    Given an actuals and prediction the class will output the evaluation metrics for a given
    """

class RegressionEvaluator:
    """
    Does the evaluations and returns the popular metrics for a regression problem such as R2, Adjusted R2 etc.
    """
@logged(logging.getLogger())
class ClassificationEvaluator:
    """
    Does the evaluations and returns the popular metrics for a classification problem such as Accuracy, F1, AUC etc.
    """

    @staticmethod
    def evaluate(actuals, pred_probs, thresholds=np.arange(0,1,0.01), maximising_metric='F1', nbins=10, pos_class=1):
        """
        Prepares a json dictionary of most of the popular classification evaluation metrics
        :return:
        """
        # Getting the predicted values for different thresholds:
        # thres_preds_df = ClassificationEvaluator.get_predictions_for_thresholds(pred_probs, thresholds)

        # Getting the evaluation metrics for different thresholds:
        eval_metrics_df = ClassificationEvaluator.get_eval_metrics_for_thresholds(actuals, pred_probs, thresholds)

        # Getting the decide data:
        deciles_df = ClassificationEvaluator.create_decile(actuals, pred_probs, pos_class=pos_class, nbins=nbins)

        # Getting the best threshold for a given metric
        best_threshold = eval_metrics_df[eval_metrics_df[maximising_metric] == max(eval_metrics_df[maximising_metric])][
            'Threshold'].reset_index(drop=True)[0]

        # Plotting the confusion matrix | precision recall curve | roc curve | class distribution:

        return eval_metrics_df, deciles_df, best_threshold

    @staticmethod
    def get_predictions_for_thresholds(pred_probs, thresholds=np.arange(0,1,0.01)):
        """
        Getting the predicted values as 0 and 1 based on different sets of threshold
        :param thresholds:
        :return:
        """
        pred_value = pd.DataFrame()
        try:
            for i in thresholds:
                col_name = "Threshold_" + str(i)
                pred_value[col_name] = [1 if j >= i else 0 for j in pred_probs]
            return pred_value

        except BaseException as e:
            logging.error('Error: ' + str(e))
            return None

    @staticmethod
    def get_eval_metrics_for_thresholds(actuals, pred_probs, thresholds=np.arange(0, 1, 0.01)):
        """
        Calculates evaluation metrics for different thresholds.
        Metrics considered =  ['TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'Recall','F1','MCC','ROC_AUC']
        :param actuals:
        :param pred_probs:
        :param thresholds:
        :return:
        """
        preds_df = ClassificationEvaluator.get_predictions_for_thresholds(pred_probs, thresholds)

        # Creating a metrics dcictionary:
        key = ['Threshold', 'TP', 'FP', 'FN', 'TN', 'Accuracy',
               'Precision0', 'Precision1', 'Recall0', 'Recall1', 'F1', 'MCC', 'ROC_AUC']
        metrics = dict([(i, []) for i in key])

        #try:
        for i in thresholds:
            col_name = "Threshold_" + str(i)
            metrics['Threshold'].append(round(i, 2))
            TN, FP, FN, TP = ClassificationEvaluator.get_confusion_matrix(actuals, preds_df[col_name])
            metrics['TP'].append(TP)
            metrics['FP'].append(FP)
            metrics['FN'].append(FN)
            metrics['TN'].append(TN)
            metrics['Accuracy'].append(accuracy_score(actuals, preds_df[col_name]))
            metrics['Precision0'].append(precision_score(actuals, preds_df[col_name], pos_label=0))
            metrics['Precision1'].append(precision_score(actuals, preds_df[col_name], pos_label=1))
            metrics['Recall0'].append(recall_score(actuals, preds_df[col_name], pos_label=0))
            metrics['Recall1'].append(recall_score(actuals, preds_df[col_name], pos_label=1))
            metrics['F1'].append(f1_score(actuals, preds_df[col_name]))
            metrics['MCC'].append(matthews_corrcoef(actuals, preds_df[col_name]))
            metrics['ROC_AUC'].append(roc_auc_score(actuals, preds_df[col_name]))

        # returning the metrics dictionary in dataframe format:
        return pd.DataFrame(metrics)

        #except BaseException as e:
        #    logging.error('Error: ' + str(e))
        #    return None

    @staticmethod
    def get_confusion_matrix(actuals, preds):
        '''
        Getting the confusion martix based on actual and predicted values
        '''
        tn, fp, fn, tp = confusion_matrix(actuals, preds).ravel()
        return(tn, fp, fn, tp)

    @staticmethod
    def create_decile(actuals, pred_probs, pos_class=1, nbins=10):
        """
        Prepares a decile table for Gain and Lift charts
        :param self:
        :param pos_class:
        :param nbins:
        :return:
        """

        valid_df = pd.DataFrame()
        valid_df['actual'] = actuals
        valid_df['p1'] = pred_probs
        # sort the probabilities
        if pos_class == 0:
            # Low probability to followed by high probability
            valid_df = valid_df.sort_values(by='p1', ascending=True)
        else:
            # High probability followed by low probability
            valid_df = valid_df.sort_values(by='p1', ascending=False)
        # splitting the probablity into bins:
        split = np.array_split(valid_df, nbins)

        col = ['decile', 'actual_true_count', 'total_count_in_decile', 'accuracy_in_decile',
               'total_true_in_population', 'true_covered', 'min_prob', 'max_prob']

        df = pd.DataFrame(columns=col)
        total_count_in_decile = []
        min_confidence = []
        max_confidence = []
        count_actual_true = []
        # Total_true_in_population=[]
        for i in split:
            total_count_in_decile.append(len(i))
            min_confidence.append(min(i['p1']))
            max_confidence.append(max(i['p1']))
            if pos_class == 0:
                count_actual_true.append(len(i[i['actual'] == 0]))
                # Plot only for the fail students
            else:
                count_actual_true.append(len(i[i['actual'] == 1]))
                # Plot only for the pass students

        df['decile'] = np.arange(1, 11)
        df['total_count_in_decile'] = total_count_in_decile
        df['min_prob'] = min_confidence
        df['max_prob'] = max_confidence
        df['actual_true_count'] = count_actual_true
        df['accuracy_in_decile'] = (df['actual_true_count'] / df['total_count_in_decile']) * 100
        df['total_true_in_population'] = sum(df['actual_true_count'])
        df['cum_sum_actual_true_count'] = df['actual_true_count'].cumsum()
        df['true_covered'] = (df['cum_sum_actual_true_count'] / df['total_true_in_population']) * 100
        df['pop'] = (df['total_count_in_decile'].cumsum() / sum(df['total_count_in_decile'])) * 100
        df['lift'] = df['true_covered'] / df['pop']
        return df