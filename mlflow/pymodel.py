import uuid
import tempfile
import logging
from autologging import logged
from sklearn.externals import joblib
import pickle
import h2o
import pandas as pd
from datetime import datetime
from helpers.evaluation_helper import ClassificationEvaluator
from helpers.common_helper import *
from utils.common import *


@logged(logging.getLogger())
class PyModel:
    """
    Wraps any python model and provides a complete metadata of the model to store it in the ModelStore
    """
    def __init__(self, model, features, split=None, feature_columns=None, target_column=None,
                 model_uuid=None, model_tag=None, model_version=None,
                 maximising_metric='F1', hyperparams=None, training_args=None,
                 train_actuals=None, train_pred_probs=None, valid_actuals=None, valid_pred_probs=None):
        """
        This init is for creation of the model
        :param model:
        :param features:
        :param split:
        :param feature_columns:
        :param target_column:
        :param model_uuid:
        :param model_tag:
        :param model_version:
        :param maximising_metric:
        :param hyperparams:
        :param training_args:
        :param train_actuals:
        :param train_pred_probs:
        :param valid_actuals:
        :param valid_pred_probs:
        """
        self._model = model
        self._features = features
        self.__log.info("Preparing the metadata.....")
        # Prepare the model metadata
        self._model_metadata = PyModelMetadata(feature_columns=feature_columns, target_column=target_column,
                                               split=split,
                                               model_uuid=model_uuid, model_tag=model_tag, model_version=model_version,
                                               maximising_metric=maximising_metric, hyperparams=hyperparams,
                                               training_args=training_args)
        self.__log.info("Collecting validation metrics...")
        # Get the training evaluation metric
        train_eval_metrics_df, train_deciles_df, train_best_threshold = ClassificationEvaluator.evaluate(
            actuals=train_actuals, pred_probs=train_pred_probs)
        self._model_metadata.training_metrics = train_eval_metrics_df.to_json()

        # Get the validation evaluation metric
        valid_eval_metrics_df, valid_deciles_df, valid_best_threshold = ClassificationEvaluator.evaluate(
            actuals=valid_actuals, pred_probs=valid_pred_probs)
        self._model_metadata.validation_metrics = valid_eval_metrics_df.to_json()

        self.__log.info("Preparing model artifacts....")
        self._create_artifacts()
        self.__log.info("Metadata successfully prepared")

    # def __init__(self, model_uuid):
    #     """
    #     This init is for loading the model from database for a model uuid
    #     :param model_uuid:
    #     """
    #     if model_uuid is None:
    #         assert False, "Model UUID must be provided"
    #
    # def __init__(self, model_tag, model_version=None):
    #     """
    #     This init is loading the model from database for a model tag and model version
    #     :param model_tag:
    #     :param model_version:
    #     """
    #     if model_tag is None:
    #         assert False, "Model Tag must be provided"
    #
    # def load_model_by_uuid(self, model_uuid):
    #     """
    #     Loads the model from database for the specified uuid
    #     :param model_uuid:
    #     :return:
    #     """
    #     model_df = pd.read_sql(
    #         'select * from {} where model_uuid = {}'.format(
    #             TBL_MODEL_STORE,
    #             model_uuid
    #         ),
    #         self.sql_client
    #     )

    @property
    def metadata(self):
        return self._model_metadata

    @metadata.setter
    def metadata(self, val):
        self._model_metadata = val

    @property
    def artifacts(self):
        return self._artifacts

    @artifacts.setter
    def artifacts(self, val):
        self._artifacts = val

    def _create_artifacts(self):
        """
        Creates serialized pickle objects and a zip it in a jar file
        :return:
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.__log.info("Saving the model artifacts to {}...".format(tmp_dir))
            # Export the model weights
            self.save_model(tmp_dir)

            # Export the feature columns
            pickle.dump(self._features, open(tmp_dir + '\\feature_columns.pkl', 'wb'))

            # zip everything to one folder
            zip_folder(tmp_dir, self._model_metadata.model_uuid)

            # set the blob to the artifact property
            self._artifacts = convert_to_binary(tmp_dir + '\\' + self._model_metadata.model_uuid + '.zip')
            self.__log.info("Successfully saved the model artifacts")

    def save_model(self, tmp_dir):
        """
        Saves the model
        :param loc:
        :return:
        """
        # Export the model weights
        joblib.dump(self._model, tmp_dir + '\\model.pkl')

    def __call__(self, *args, **kwargs):
        return self.to_df()

    def to_df(self):
        """
        Returns the Metadata as a Pandas dataframe
        :return:
        """
        # data = [{"model_uuid": self._model_metadata._model_uuid, "model_tag": self._model_metadata._model_tag,
        #          "model_version": self._model_metadata._model_version,
        #          "feature_columns": self._model_metadata._feature_columns,
        #          "target_column": self._model_metadata._target_column,
        #          "training_args": self._model_metadata._training_args,
        #          "hyperparams": self._model_metadata._hyperparams, "split": self._model_metadata._split,
        #          "training_metrics": self._model_metadata._training_metrics,
        #          "validation_metrics": self._model_metadata._validation_metrics}]
        # meta_df = pd.DataFrame(data)
        # return meta_df
        return self._model_metadata()

    def describe(self, verbose=False, show_plots=True):
        """
        Describes a model
        :return:
        """
        # Get
        return self.to_df()


class PyH20Model(PyModel):
    """
    Extends H20 Model. Only difference is in H2O model is saved as POJO and Jar
    """

    def __init__(self, *args):
        PyModel.__init__(self, *args)

    def save_model(self, tmp_dir):
        """
        Saves the model
        :param loc:
        :return:
        """
        # Downloading the pojo and the JAR file
        try:
            h2o.download_pojo(self._model, path = tmp_dir, get_jar = True)
        except Exception as e:
            #logger.error('Failed to save the model: '+ str(e))
            pass


class PyModelMetadata:
    """
    Stores all the metadata of a model
    """
    def __init__(self, feature_columns=None, target_column=None, split=None,
                 model_uuid=None, model_tag=None, model_version=None, maximising_metric='f1',
                 hyperparams=None, training_args=None, training_metrics=None, validation_metrics=None):
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._split = split
        self._model_uuid = model_uuid
        self._model_tag = model_tag
        self._model_version = model_version
        self._maximizing_metric = maximising_metric
        self._hyperparams = hyperparams
        self._training_args = training_args
        self._training_metrics = training_metrics
        self._validation_metrics = validation_metrics
        self._created_by = get_username3()
        self._created_date = datetime.utcnow()
        self._rundate = self._created_date

    def _get_uuid(self):
        """
        Generates a uuid if not present
        :return:
        """
        if self._model_uuid is None:
            self._model_uuid = uuid.uuid1()

    @property
    def feature_columns(self):
        return self._feature_columns

    @feature_columns.setter
    def feature_columns(self, val):
        self._feature_columns = val

    @property
    def target_column(self):
        return self._target_column

    @target_column.setter
    def target_column(self, val):
        self._target_column = val

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, val):
        self._split = val

    @property
    def model_uuid(self):
        return self._model_uuid

    @model_uuid.setter
    def model_uuid(self, val):
        self._model_uuid = val

    @property
    def model_tag(self):
        return self._model_tag

    @model_tag.setter
    def model_tag(self, val):
        self._model_tag = val

    @property
    def model_version(self):
        return self._model_version

    @model_version.setter
    def model_version(self, val):
        self._model_version = val

    @property
    def hyperparams(self):
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, val):
        self._hyperparams = val

    @property
    def maximizing_metric(self):
        return self._maximizing_metric

    @maximizing_metric.setter
    def maximizing_metric(self, val):
        self._maximizing_metric = val

    @property
    def training_args(self):
        return self._training_args

    @training_args.setter
    def training_args(self, val):
        self._training_args = val

    @property
    def training_metrics(self):
        return self._training_metrics

    @training_metrics.setter
    def training_metrics(self, val):
        self._training_metrics = val

    @property
    def validation_metrics(self):
        return self._validation_metrics

    @validation_metrics.setter
    def validation_metrics(self, val):
        self._validation_metrics = val

    @property
    def training_start_time(self):
        return self._training_start_time

    @training_start_time.setter
    def training_start_time(self, val):
        self._training_start_time = val

    @property
    def training_end_time(self):
        return self._training_end_time

    @training_end_time.setter
    def training_end_time(self, val):
        self._training_end_time = val

    @property
    def total_training_time(self):
        return self._total_training_time

    @total_training_time.setter
    def total_training_time(self, val):
        self._total_training_time = val

    @property
    def trained_by(self):
        return self._trained_by

    @trained_by.setter
    def trained_by(self, val):
        self._trained_by = val

    @property
    def seed_value(self):
        return self._seed_value

    @seed_value.setter
    def seed_value(self, val):
        self._seed_value = val

    @property
    def training_data_reference(self):
        return self._training_data_reference

    @training_data_reference.setter
    def training_data_reference(self, val):
        self._training_data_reference = val

    @property
    def test_data_reference(self):
        return self._test_data_reference

    @test_data_reference.setter
    def test_data_reference(self, val):
        self._test_data_reference = val

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, val):
        self._model_path = val

    @property
    def feature_importance(self):
        return self._feature_importance

    @feature_importance.setter
    def feature_importance(self, val):
        self._feature_importance = val

    @property
    def feature_distribution(self):
        return self._feature_distribution

    @feature_distribution.setter
    def feature_distribution(self, val):
        self._feature_distribution = val

    @property
    def training_count(self):
        return self._training_count

    @training_count.setter
    def training_count(self, val):
        self._training_count = val

    @property
    def validation_count(self):
        return self._validation_count

    @validation_count.setter
    def validation_count(self, val):
        self._validation_count = val

    @property
    def total_count(self):
        return self._total_count

    @total_count.setter
    def total_count(self, val):
        self._total_count = val

    def __call__(self, *args, **kwargs):
        return self.to_df()

    def to_df(self):
        """
        Returns the Metadata as a Pandas dataframe
        :return:
        """
        data = [{"model_uuid": self._model_uuid, "model_tag": self._model_tag, "model_version": self._model_version,
                 "feature_columns": self._feature_columns, "target_column": self._target_column,
                 "training_args": self._training_args, "hyperparams": self._hyperparams, "split": self._split,
                 "training_metrics": self._training_metrics, "validation_metrics": self._validation_metrics,
                 "training_start_time": self._training_start_time, "training_end_time": self._training_end_time,
                 "total_training_time": self._total_training_time, "feature_importance": self._feature_importance,
                 "feature_distribution": self._feature_distribution, "training_count": self._training_count,
                 "validation_count": self._validation_count, "total_count": self._total_count,
                 "seed_value": self._seed_value, "rundate": self._rundate, "trained_by": self._created_by,
                 "created_date": self._rundate, "created_by": self._created_by}]
        meta_df = pd.DataFrame(data)
        return meta_df

    def to_table(self):
        """
        Transforms the meta data to database table format for db operations
        :return:
        """
        data = [{"Model_UUID": self._model_uuid, "Model_Tag": self._model_tag, "Model_Version": self._model_version,
                 "Input_Features": str(self._feature_columns), "Target_Column": str(self._target_column),
                 "Training_Args": str(self._training_args), "Hyperparams": str(self._hyperparams), "Split": self._split,
                 "Training_Metrics": str(self._training_metrics), "Validation_Metrics": str(self._validation_metrics),
                 "Training_Start_Time": self._training_start_time, "Training_End_Time": self._training_end_time,
                 "Total_Training_Time": self._total_training_time, "Feature_Importance": self._feature_importance,
                 "Feature_Distribution": self._feature_distribution, "Training_Count": self._training_count,
                 "Validation_Count": self._validation_count, "Total_Count": self._total_count,
                 "Seed_Value": self._seed_value, "Rundate": self._rundate, "Trained_By": self._created_by,
                 "Created_Date": self._rundate, "Created_By": self._created_by}]
        meta_df = pd.DataFrame(data)
        return meta_df