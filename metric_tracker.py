from typing import List

import numpy as np
import omegaconf
import pandas as pd

from metrics import calculate_confusion_matrix, calculate_classification_metrics, calculate_regression_metrics
from util.time.timestamp import Timestamp


class DictConfig(omegaconf.DictConfig):
    """
    Class to inform arguments needed for MetricTracker.
    """
    evaluation_thresholds: List[float]


class MetricTracker:
    """
    Class to track and save metrics for regional and grid predictions.

    Attributes:
        evaluation_thresholds: list of thresholds for classification
        binary_classification_index: index of binary threshold for binary classification metrics
        regional_df: raw data for regional predictions.
        grid_df: statistics (confusion matrix) for grid predictions aggregated over timestamps and regions.
    """

    def __init__(
        self,
        args: DictConfig
    ) -> None:
        self.evaluation_thresholds = args.evaluation_thresholds
        self.binary_classification_index = 2  # index of value 나쁨 # TODO  # NAMGYU: do what?

        self.regional_df = pd.DataFrame(
            columns=['origin', 'lead_time', 'region', 'target', 'prediction', 'categorical_target',
                     'categorical_prediction'])
        self.grid_df = pd.DataFrame(
            columns=['origin', 'lead_time', 'region', 
                     't0p0', 't0p1', 't0p2', 't0p3',
                     't1p0', 't1p1', 't1p2', 't1p3',
                     't2p0', 't2p1', 't2p2', 't2p3',
                     't3p0', 't3p1', 't3p2', 't3p3'])

    def add_regional_predictions(self, timestamp: List[Timestamp], prediction: np.array,
                                 target: np.array):
        """
        Update self.regional_df with new batch of regional predictions (and targets), given in continuous values.

        Args:
            timestamp: list of timestamps
            prediction: np.array of shape (B, N) with continuous values
            target: np.array of shape (B, N) with continuous values

        Returns:
        """

        categorical_prediction = np.digitize(prediction, bins=self.evaluation_thresholds)
        categorical_target = np.digitize(target, bins=self.evaluation_thresholds)

        # save into dataframe
        b = len(timestamp)
        origins = [t.origin for t in timestamp for _ in range(19)]
        lead_times = [t.lead_time for t in timestamp for _ in range(19)]
        regions = list(range(19)) * b

        new_data = pd.DataFrame({
            'origin': origins,
            'lead_time': lead_times,
            'region': regions,
            'target': target.flatten().tolist(),
            'prediction': prediction.flatten().tolist(),
            'categorical_target': categorical_target.flatten().tolist(),
            'categorical_prediction': categorical_prediction.flatten().tolist()
        })
        self.regional_df = pd.concat([self.regional_df, new_data], ignore_index=True)

    def add_grid_predictions(self, timestamp: List[Timestamp], prediction: np.array,
                             target: np.array) -> None:
        """
        Update self.grid_df with new batch of grid predictions (and targets), given in continuous values.

        :param timestamp: list of B timestamps
        :param prediction: np.array of shape (B, W, H)
        :param target: np.array of shape (B, W, H)
        """
        b = len(timestamp)
        prediction = prediction.reshape(b, -1)
        target = target.reshape(b, -1)

        categorical_prediction = np.digitize(prediction, bins=self.evaluation_thresholds)
        categorical_target = np.digitize(target, bins=self.evaluation_thresholds)

        # load regional masks into shape of (19, -1)
        masks = np.random.randint(2, size=(19, 83, 67))  # TODO GET_MASK_HERE # (M, W, H) # TODO -1 for missing value
        masks = masks.reshape(19, -1)

        # split the prediction and target into 19 regions
        categorical_prediction = np.repeat(categorical_prediction[:, np.newaxis, :], 19, axis=1)  # (B, R, W*H)
        categorical_target = categorical_target[:, np.newaxis, :] * masks[np.newaxis, :, :]  # (B, R, W*H)

        # calculate confusion matrix    
        classification_output = calculate_confusion_matrix(categorical_prediction.reshape(b * 19, -1),
                                                           categorical_target.reshape(b * 19, -1),
                                                           n_classes=len(self.evaluation_thresholds) + 1)  # (B*R, 4*4)

        # save into dataframe
        origins = [t.origin for t in timestamp for _ in range(19)]
        lead_times = [t.lead_time for t in timestamp for _ in range(19)]
        regions = list(range(19)) * b

        classification_output = classification_output.reshape(b, 19, 4, 4)

        new_data = pd.DataFrame({
            'origin': origins,
            'lead_time': lead_times,
            'region': regions,
            't0p0': classification_output[:, :, 0, 0].flatten().tolist(),
            't0p1': classification_output[:, :, 0, 1].flatten().tolist(),
            't0p2': classification_output[:, :, 0, 2].flatten().tolist(),
            't0p3': classification_output[:, :, 0, 3].flatten().tolist(),
            't1p0': classification_output[:, :, 1, 0].flatten().tolist(),
            't1p1': classification_output[:, :, 1, 1].flatten().tolist(),
            't1p2': classification_output[:, :, 1, 2].flatten().tolist(),
            't1p3': classification_output[:, :, 1, 3].flatten().tolist(),
            't2p0': classification_output[:, :, 2, 0].flatten().tolist(),
            't2p1': classification_output[:, :, 2, 1].flatten().tolist(),
            't2p2': classification_output[:, :, 2, 2].flatten().tolist(),
            't2p3': classification_output[:, :, 2, 3].flatten().tolist(),
            't3p0': classification_output[:, :, 3, 0].flatten().tolist(),
            't3p1': classification_output[:, :, 3, 1].flatten().tolist(),
            't3p2': classification_output[:, :, 3, 2].flatten().tolist(),
            't3p3': classification_output[:, :, 3, 3].flatten().tolist()
        })  # NAMGYU: let's shorten this with a for-loop
        self.grid_df = pd.concat([self.grid_df, new_data], ignore_index=True)

    def print_metrics(self):
        """
        Print regression and classification metrics based on self.regional_df
        """
        # REGIONAL
        # regression
        reg_m = calculate_regression_metrics(np.array(self.regional_df['categorical_prediction']), 
                                             np.array(self.regional_df['categorical_target']))
        print(f"nmb: {round(reg_m['nmb'], 2)}, nme: {round(reg_m['nme'], 2)}, r: {round(reg_m['r'], 2)}, rmse: {round(reg_m['rmse'], 2)}")

        # classification
        categorical_predictions = np.array(self.regional_df['categorical_prediction']).reshape(-1, 1).astype(int)
        categorical_targets = np.array(self.regional_df['categorical_target']).reshape(-1, 1).astype(int)
        cm = calculate_confusion_matrix(categorical_predictions, categorical_targets, n_classes=len(self.evaluation_thresholds)+1)
        cat_m = calculate_classification_metrics(cm, 
                                                 binary_classification_index=self.binary_classification_index)
        print(f"acc: {round(cat_m['acc'], 2)}, hard_acc: {round(cat_m['hard_acc'], 2)}, far: {round(cat_m['far'], 2)}, pod: {round(cat_m['pod'], 2)}, f1: {round(cat_m['f1'], 2)}")

        # GRID
        # classification
        columns_to_convert = ['t0p0', 't0p1', 't0p2', 't0p3', 
                              't1p0', 't1p1', 't1p2', 't1p3', 
                              't2p0', 't2p1', 't2p2', 't2p3', 
                              't3p0', 't3p1', 't3p2', 't3p3']
        grid_df = self.grid_df[columns_to_convert].to_numpy().reshape(-1, 4, 4)
        cat_m = calculate_classification_metrics(grid_df, 
                                                 binary_classification_index=self.binary_classification_index)
        print(f"acc: {round(cat_m['acc'], 2)}, hard_acc: {round(cat_m['hard_acc'], 2)}, far: {round(cat_m['far'], 2)}, pod: {round(cat_m['pod'], 2)}, f1: {round(cat_m['f1'], 2)}")
        
