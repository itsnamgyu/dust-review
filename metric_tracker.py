from typing import List

import numpy as np
import omegaconf
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from metrics import calculate_confusion_matrix
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
            columns=['origin', 'lead_time', 'region', 'g0p0', 'g0p1', 'g0p2', 'g0p3', 'g1p0', 'g1p1', 'g1p2', 'g1p3',
                     'g2p0', 'g2p1', 'g2p2', 'g2p3', 'g3p0', 'g3p1', 'g3p2', 'g3p3'])

        self.device = args.device

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
                                                           n_classes=len(self.evaluation_thresholds) + 1,
                                                           device=self.device)  # (B*R, 4*4)

        # save into dataframe
        origins = [t.origin for t in timestamp for _ in range(19)]
        lead_times = [t.lead_time for t in timestamp for _ in range(19)]
        regions = list(range(19)) * b

        classification_output = classification_output.reshape(b, 19, 4, 4)

        new_data = pd.DataFrame({
            'origin': origins,
            'lead_time': lead_times,
            'region': regions,
            'g0p0': classification_output[:, :, 0, 0].flatten().tolist(),
            'g0p1': classification_output[:, :, 0, 1].flatten().tolist(),
            'g0p2': classification_output[:, :, 0, 2].flatten().tolist(),
            'g0p3': classification_output[:, :, 0, 3].flatten().tolist(),
            'g1p0': classification_output[:, :, 1, 0].flatten().tolist(),
            'g1p1': classification_output[:, :, 1, 1].flatten().tolist(),
            'g1p2': classification_output[:, :, 1, 2].flatten().tolist(),
            'g1p3': classification_output[:, :, 1, 3].flatten().tolist(),
            'g2p0': classification_output[:, :, 2, 0].flatten().tolist(),
            'g2p1': classification_output[:, :, 2, 1].flatten().tolist(),
            'g2p2': classification_output[:, :, 2, 2].flatten().tolist(),
            'g2p3': classification_output[:, :, 2, 3].flatten().tolist(),
            'g3p0': classification_output[:, :, 3, 0].flatten().tolist(),
            'g3p1': classification_output[:, :, 3, 1].flatten().tolist(),
            'g3p2': classification_output[:, :, 3, 2].flatten().tolist(),
            'g3p3': classification_output[:, :, 3, 3].flatten().tolist()
        })  # NAMGYU: let's shorten this with a for-loop
        self.grid_df = pd.concat([self.grid_df, new_data], ignore_index=True)

    def print_regional_metrics(self):
        """
        Print regression and classification metrics based on self.regional_df
        """
        # regression
        raw_data = self.regional_df

        target = np.array(raw_data['target'])
        predictions = np.array(raw_data['prediction'])

        nmb = np.sum(predictions - target) / np.sum(target)
        nme = np.sum(np.abs(predictions - target)) / np.sum(np.abs(target))
        r, _ = pearsonr(predictions, target)
        rmse = np.sqrt(mean_squared_error(target, predictions))

        print(f"nmb: {round(nmb, 2)}, nme: {round(nme, 2)}, r: {round(r, 2)}, rmse: {round(rmse, 2)}")

        # classification
        categorical_predictiones = np.array(raw_data['categorical_prediction']).reshape(-1, 1).astype(
            int)  # reshape(1, -1)
        categorical_targetes = np.array(raw_data['categorical_target']).reshape(-1, 1).astype(int)  # reshape(1, -1)
        confusion_matrix = calculate_confusion_matrix(categorical_predictiones, categorical_targetes,
                                                      n_classes=len(self.evaluation_thresholds) + 1,
                                                      device=self.device)

        self.print_categorical_metrics(confusion_matrix)

    def print_categorical_metrics(self, cm: np.array, binary_classification_index: int = 2):
        """
        Print classification metrics for confusion matrix (self.grid_df and self.regional_df)

        Args:
            cm: confusion matrix. ndarray of shape (n_samples, n_classes, n_classes)
        """

        """
        :param cm:
        """
        # classification
        bindex = binary_classification_index
        tn = np.sum(cm[:, :bindex, :bindex])
        fp = np.sum(cm[:, :bindex, bindex:])
        fn = np.sum(cm[:, bindex:, :bindex])
        tp = np.sum(cm[:, bindex:, bindex:])
        total = np.sum(cm)

        acc = np.sum(cm[:, 0, 0] + cm[:, 1, 1] + cm[:, 2, 2] + cm[:, 3, 3]) / total
        hard_acc = np.sum(cm[:, 2, 2] + cm[:, 3, 3]) / (fn + tp)
        far = fp / (fp + tp)
        pod = tp / (tp + fn)
        f1 = 2 * pod * (1 - far) / (pod + (1 - far))

        print(f"acc: {acc:6.{2}f}, hard_acc: {hard_acc:6.{2}f}, far: 6{far:.{2}f}")
        print(f"pod: {pod:6.{2}f}, f1: {f1:6.{2}f}")
