import numpy as np
import numpy.ma as ma
import omegaconf
from typing import Set, Dict
import pandas as pd

from util.time.timestamp import Timestamp
from util.convert import categorial_to_continuous, continuous_to_categorical

import torch


class DictConfig(omegaconf.DictConfig):
    n_regions: int



class MetricTracker:
    def __init__(
        self,
        args: DictConfig
    ) -> None:
        n_regression_metrics = 4  # NAMGYU: what is this?
        # NAMGYU: I think n_xxx or xxx_count is a more standard name than xxx_num
        # region_num = 19  # NAMGYU: is this needed?
        # leadtime_num = 67  # NAMGYU: is this needed?
        # grid_num = 67 * 82  # NAMGYU: is this needed?

        self.evaluation_thresholds = args.evaluation_threhsolds

        # NAMGYU: I think these names are unclear
        # NAMGYU: I don't know why these are in this format. Let's discuss
        # self.regional_regression_output = np.zeros((n_regression_metrics))  # / self.total_num
        # self.regional_classification_output = np.zeros((4, 4))  # / self.total_num
        # self.grid_classification_output = np.zeros((4, 4))  # / self.total_num

        # NAMGYU: these names are unclear
        # self.regional_num = 0
        # self.grid_num = 0

        self.regional_df = pd.DataFrame(columns=['time', 'region', 'GT', 'pred'])
        self.regional_df = pd.DataFrame(columns=['time', 'region', 'ground_truth', 'prediction'])
        # NAMGYU: fixed to use full wods

        self.grid_df = pd.DataFrame(columns=['time', 'region', 'conf'])
        # NAMGYU: please use full words here as well

        self.regression_to_idx = {
            'nmb': 0,
            'nme': 1,
            'r': 2,
            'rmse': 3
        }

    def add_regional_predictions(self, timestamp: list[Timestamp], prediction: np.array,
                                 ground_truth: np.array):  # continuous input -> regression, classification
        # NAMGYU: I think all variable names should be plural (since they are / (can be) batched)
        """
        NAMGYU: can you add some explanations / array shapes?


        :param timestamp:
        :param prediction:
        :param ground_truth:
        :return:
        """

        leadtime_idx = np.array([t.lead_time for t in timestamp]) - 6

        # regression
        # NAMGYU: does leadtime index need to go here?
        # NAMGYU: do these need to be functions?
        # NAMGYU: the function names are misleading. these functions calculate the metrics and then call
        #         add_metrics. I think this process is too complicated
        self.calculate_nmb(prediction, ground_truth, leadtime_idx)
        self.calculate_nme(prediction, ground_truth, leadtime_idx)
        self.calculate_r(prediction, ground_truth, leadtime_idx)
        self.calculate_rmse(prediction, ground_truth, leadtime_idx)

        # convert into categorical
        categorical_prediction = continuous_to_categorical(prediction, self.evaluation_thresholds)
        categorical_ground_truth = continuous_to_categorical(ground_truth, self.evaluation_thresholds)

        # classification
        classification_output = self.calculate_confusion_matrix(categorical_prediction, categorical_ground_truth)
        self.regional_classification_output += classification_output

        # save into dataframe
        timestamp = [t.to_datetime() for t in timestamp] * 19
        self.regional_df = self.regional_df.append(
            # NAMGYU: changed the column names to full words
            # NAMGYU: you should name pandas columns like you would name variables
            pd.DataFrame({
                'timestamp': timestamp,
                'region': list(range(1, 20)) * len(timestamp),  # NAMGYU: are regions 1-indexed?
                'ground_truth': ground_truth.tolist(),
                'prediction': prediction.tolist(),
                'ground_truth_class': categorical_ground_truth.tolist(),
                'prediction_class': categorical_prediction.tolist()
            }))

    def add_grid_predictions(self, timestamp: list[Timestamp], prediction: np.array,
                             ground_truth: np.array):  # continuous input -> classification
        """
        NAMGYU: can you add some explanations / array shapes?
        :param timestamp:
        :param prediction:
        :param ground_truth:
        :return:
        """
        b = len(timestamp)
        prediction = prediction.reshape(b, -1)
        ground_truth = ground_truth.reshape(b, -1)

        # convert into categorical  # NAMGYU: this comment is unnecessary (the method name is self-explanatory - this is good)
        categorical_prediction = continuous_to_categorical(prediction, self.evaluation_thresholds)

        # classification
        classification_output = self.calculate_confusion_matrix(categorical_prediction, ground_truth)
        self.grid_classification_output += classification_output

        # save into data

    def save_df(self, path: str) -> None:
        self.regional_df.to_csv(path, index=False)

    def print_result(self) -> Dict:
        # average the metrics and print the regional result
        metrics = {}
        metrics['nmb'] = self.regional_regression_output[self.regression_to_idx['nmb']] / self.grid_num
        metrics['nme'] = self.regional_regression_output[self.regression_to_idx['nme']] / self.grid_num
        metrics['r'] = self.regional_regression_output[self.regression_to_idx['r']] / self.grid_num
        metrics['rmse'] = self.regional_regression_output[self.regression_to_idx['rmse']] / self.grid_num

        metrics['acc'] = sum(self.regional_classification_output[i, i] for i in range(4)).sum() / np.sum(self.grid_num)
        metrics['acc_hard'] = np.sum(
            self.regional_classification_output[2, 2] + self.regional_classification_output[3, 3]) / np.sum(
            self.regional_classification_output[2:, :])
        metrics['far'] = np.sum(self.regional_classification_output[:2, 2:]) / np.sum(
            self.regional_classification_output[:, 2:])
        metrics['pod'] = np.sum(self.regional_classification_output[2:, 2:]) / np.sum(
            self.regional_classification_output[2:, :])
        metrics['f1'] = 2 * metrics['pod'] * (1 - metrics['far']) / (metrics['pod'] + (1 - metrics['far']))

        print(f"Regional result: {metrics}")

        # average the metrics and print the grid result
        metrics = {}
        metrics['acc'] = sum(self.grid_classification_output[i, i] for i in range(4)).sum() / np.sum(self.grid_num)
        metrics['acc_hard'] = np.sum(
            self.grid_classification_output[2, 2] + self.grid_classification_output[3, 3]) / np.sum(
            self.grid_classification_output[2:, :])
        metrics['far'] = np.sum(self.grid_classification_output[:2, 2:]) / np.sum(
            self.grid_classification_output[:, 2:])
        metrics['pod'] = np.sum(self.grid_classification_output[2:, 2:]) / np.sum(
            self.grid_classification_output[2:, :])
        metrics['f1'] = 2 * metrics['pod'] * (1 - metrics['far']) / (metrics['pod'] + (1 - metrics['far']))

        print(f"Grid result: {metrics}")

    #######################################################
    # metric calculation
    #######################################################

    def add_metrics(self, metric_idx: list, metric_result: np.array, leadtime_idx: list) -> None:
        self.regional_regression_output[metric_idx, leadtime_idx] += metric_result

    def calculate_nmb(self, output: np.array, target: np.array, leadtime_idx: list) -> np.array:
        nmb = np.sum(output - target, axis=1) / np.sum(target, axis=1)
        self.add_metrics(self.regression_to_idx['nmb'], nmb, leadtime_idx)

    def calculate_nme(self, output: np.array, target: np.array, leadtime_idx: list) -> np.array:
        nme = np.sum(np.abs(output - target), axis=1) / np.sum(target, axis=1)
        self.add_metrics(self.regression_to_idx['nme'], nme, leadtime_idx)


    def calculate_r(self, output: np.array, target: np.array, leadtime_idx: list) -> np.array:
        r = np.sum(output * target, axis=1) / np.sqrt(np.sum(output ** 2, axis=1) * np.sum(target ** 2, axis=1))
        self.add_metrics(self.regression_to_idx['r'], r, leadtime_idx)

    def calculate_rmse(self, output: np.array, target: np.array, leadtime_idx: list) -> np.array:
        rmse = np.sqrt(np.sum((output - target) ** 2, axis=1) / target.shape[1])
        self.add_metrics(self.regression_to_idx['rmse'], rmse, leadtime_idx)

    def calculate_confusion_matrix(self, output: np.array, target: np.array, leadtime_idx: list = None) -> np.array:
        B, _ = output.shape
        classification_output = np.zeros((B, 4, 4), dtype=int)

        pred_indices = output[:, :, None]  # Shape (B, N, 1)
        gt_indices = target[:, None, :]  # Shape (B, 1, N)

        # Broadcast batch indices
        batch_indices = np.arange(B)[:, None, None]  # Shape (B, 1, 1)

        # Use numpy's add.at to accumulate counts in the result matrix
        np.add.at(classification_output, (batch_indices, pred_indices, gt_indices), 1)

        return classification_output
