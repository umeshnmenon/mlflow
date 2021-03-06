import numpy as np
import pandas as pd
import logging
from autologging import logged
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,average_precision_score, recall_score, f1_score, roc_curve ,auc, matthews_corrcoef, roc_auc_score, classification_report


@logged(logging.getLogger())
class ClassificationEvaluator:
    """
    Does the evaluations and returns the popular metrics for a classification problem such as Accuracy, F1, AUC etc.
    """

    def __init__(self, actuals, pred_probs, thresholds=np.arange(0,1,0.01), maximising_metric='F1', nbins=10, pos_class=1):
        """
        Moving away with static class to use seamlessly with plot class
        """
        self.actuals = actuals
        self.pred_probs = pred_probs
        self.thresholds = thresholds
        self.maximising_metric = maximising_metric
        self.nbins = nbins
        self.pos_class = pos_class

    def evaluate(self, actuals, pred_probs, thresholds=np.arange(0,1,0.01), maximising_metric='F1', nbins=10, pos_class=1):
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
    def get_predictions_for_thresholds(pred_probs, thresholds=np.arange(0, 1, 0.01)):
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

    @staticmethod
    def show_plots():
        """
        Shows all the evaluation metric plots
        :return:
        """

    class ClassificationEvaluationPlot:
        """
        This class creates the plots for classification evaluation metrics. This class cannot work as standalone. It must be
        called from ClassificationEvaluator class.
        """

        def __init__(self, metrics):
            """
            Stored the metrics and creates plots based on the metrics
            :param metrics:
            """
            self.metrics = metrics

        def plot_metrics(self):
            '''
            Plotting the graphs for all|specified evaluation metrics
            '''
            for i in self.metrics:
                sns.lineplot(x='Threshold', y=i, hue=self.hue_feat, markers=True, dashes=True, data=self.metrics_db)
                plt.show()

        def metric_plots_2(self, actual=None, pred=None, threshold=0.5):
            '''
            TODO : due to package issue this function is not used (need to check)
            Plotting the roc Curve, Precision recall curve, confusion matirx
            '''
            if ((actual is not None) & (pred is not None)):
                bc = BinaryClassification(y_true=actual, y_pred=pred, labels=["Class 0", "Class 1"],
                                          threshold=threshold)
                # Figures
                plt.figure(figsize=(20, 15))
                plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
                # Roc curve:
                bc.plot_roc_curve(threshold=threshold)
                plt.subplot2grid((2, 6), (0, 2), colspan=2)
                # precision recall curve:
                bc.plot_precision_recall_curve(threshold=threshold)
                plt.subplot2grid((2, 6), (0, 4), colspan=2)
                # class distribution curve:
                bc.plot_class_distribution(threshold=threshold)
                plt.subplot2grid((2, 6), (1, 1), colspan=2)
                # confusion matrix:
                bc.plot_confusion_matrix(threshold=threshold)
                plt.subplot2grid((2, 6), (1, 3), colspan=2)
                # normalised confusion matrix:
                bc.plot_confusion_matrix(threshold=threshold, normalize=True)
                plt.show()
                # Classification report:
                print(classification_report(actual, [0 if i <= threshold else 1 for i in pred]))

        def classification_report_chart(self, actual=None, pred=None, best_threshold=0.5):
            # printing the classification report:
            print(classification_report(actual, [0 if i <= best_threshold else 1 for i in pred]))

        def roc_auc_plot(self, actual=None, pred=None):
            # Preparing ROC curve
            false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # Plotting ROC Curve
            plt.title('ROC CURVE')
            plt.plot(false_positive_rate, true_positive_rate, 'b',
                     label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.2])
            plt.ylim([-0.1, 1.2])
            plt.ylabel('True_Positive_Rate')
            plt.xlabel('False_Positive_Rate')
            plt.show()

        def precision_recall_plot(self, actual=None, pred=None):
            precision, recall, _ = precision_recall_curve(actual, pred)
            average_precision = average_precision_score(actual, pred)
            # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
            step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve: Avg Precision ={0:0.2f}'.format(average_precision))

        def confusion_matrix_plot(self, actual=None, pred=None, best_threshold=0.5):
            '''
            Plotting the Confusion matrix based on the best threshold
            '''
            if ((actual is not None) & (pred is not None)):
                pred_value = [1 if i >= best_threshold else 0 for i in pred]
                cm = confusion_matrix(actual, pred_value)
                plt.clf()
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.gray_r)
                classNames = ['Negative', 'Positive']
                plt_title = "Confusion Matrix plot - Threshold (" + str(best_threshold) + ")"
                plt.title(plt_title)
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                tick_marks = np.arange(len(classNames))
                plt.xticks(tick_marks, classNames, rotation=45)
                plt.yticks(tick_marks, classNames)
                s = [['TN', 'FP'], ['FN', 'TP']]
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]), bbox=dict(facecolor='white', alpha=0.5))
                plt.show()

        def plot_decile(self, df=None, model_class=0):
            '''
            Plotting the decile chart
            '''
            plt.style.use('seaborn-pastel')
            f, (ax1) = plt.subplots(1, figsize=(14, 7))
            # fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            if model_class == 0:
                ax1.bar(x='Pop', height='Accuracy_in_decile', width=6, color='red', data=df, label="Accuracy_in_decile")
            else:
                ax1.bar(x='Pop', height='Accuracy_in_decile', width=6, color='green', data=df,
                        label="Accuracy_in_decile")

            ax2.plot('Pop', 'True_covered', data=df, linestyle='-', color='black', marker='o', label="True_covered")
            for index, row in df.iterrows():
                if row.name < 3:
                    ax1.text((row.name + 0.9) * 12, row['Accuracy_in_decile'],
                             str(round(row['Accuracy_in_decile'], 2)) + '%',
                             color='black', ha="right", fontsize=10)
                    ax2.text((row.name + 0.6) * 11, row['True_covered'], str(round(row['True_covered'], 2)) + '%',
                             color='black', ha="center", fontsize=10)
                elif row.name >= 3 and row.name < 6:
                    ax1.text((row.name + 0.5) * 12, row['Accuracy_in_decile'],
                             str(round(row['Accuracy_in_decile'], 2)) + '%',
                             color='black', ha="right", fontsize=10)
                    ax2.text((row.name + 0.1) * 11, row['True_covered'], str(round(row['True_covered'], 2)) + '%',
                             color='black', ha="center", fontsize=10)
                else:
                    ax1.text((row.name - 0.2) * 12, row['Accuracy_in_decile'],
                             str(round(row['Accuracy_in_decile'], 2)) + '%',
                             color='black', ha="right", fontsize=10)
                    ax2.text((row.name - 0.1) * 11, row['True_covered'], str(round(row['True_covered'], 2)) + '%',
                             color='black', ha="center", fontsize=10)
            ax2.grid(b=False)
            ax1.set_title('Lift Chart')
            ax1.set_ylabel('Accuracy in Decile')
            ax2.set_ylabel('True Covered')
            ax1.set_xlabel('Cumulative Population')
            plt.show()

        def lift_chart(self, data=None, X='decile', y='lift'):
            '''
            Plotting the lift chart against the deciles
            '''
            if data is not None:
                sns.lineplot(x=X, y=y, markers=True, dashes=True, data=data)
                plt.show()


@logged(logging.getLogger())
class ClassificationEvaluator1:
    """
    Does the evaluations and returns the popular metrics for a classification problem such as Accuracy, F1, AUC etc.
    Static class
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

    @staticmethod
    def show_plots():
        """
        Shows all the evaluation metric plots
        :return:
        """

