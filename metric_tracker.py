import numpy as np
import numpy.ma as ma
import omegaconf
from typing import Set, Dict, List
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from util.time.timestamp import Timestamp

from metrics import calculate_confusion_matrix

class DictConfig(omegaconf.DictConfig):
    n_regions: int



class MetricTracker:
    def __init__(
        self,
        args: DictConfig
    ) -> None:
        # n_regression_metrics = 4
        self.evaluation_thresholds = args.evaluation_threhsolds
        self.evaluation_boundary = 2 # index of value 나쁨 # TODO

        self.regional_df = pd.DataFrame(columns=['date', 'leadtime', 'region', 'target_cont', 'predict_cont', 'target_cate', 'predict_cate'])
        self.grid_df = pd.DataFrame(columns=['date', 'leadtime', 'region', 'g0p0', 'g0p1', 'g0p2', 'g0p3', 'g1p0', 'g1p1', 'g1p2', 'g1p3', 'g2p0', 'g2p1', 'g2p2', 'g2p3', 'g3p0', 'g3p1', 'g3p2', 'g3p3'])

        self.device = args.device

    def add_regional_predictions(self, timestamp: List[Timestamp], prediction: np.array, target: np.array):  # continuous input -> regression, classification
        """
        Add raw regional predictions to self.regional_df

        :param timestamp: list of timestamps
        :param prediction: np.array of shape (B, N)
        :param target: np.array of shape (B, N)
        """

        categorical_prediction = np.digitize(prediction, bins=self.evaluation_thresholds)
        categorical_target = np.digitize(target, bins=self.evaluation_thresholds)

        # save into dataframe
        b = len(timestamp)
        dates = [t.origin for t in timestamp for _ in range(19)]
        lead_times = [t.lead_time for t in timestamp for _ in range(19)]
        regions = list(range(19)) * b

        new_data = pd.DataFrame({
            'date': dates,
            'leadtime': lead_times,
            'region': regions, 
            'target_cont': target.flatten().tolist(),
            'predict_cont': prediction.flatten().tolist(),
            'target_cate': categorical_target.flatten().tolist(),
            'predict_cate': categorical_prediction.flatten().tolist()
        })
        self.regional_df = pd.concat([self.regional_df, new_data], ignore_index=True)

    def add_grid_predictions(self, timestamp: List[Timestamp], prediction: np.array, target: np.array):  # continuous input -> classification
        """
        Add confusion matrix of grid predictions to self.grid_df

        :param timestamp: list of timestamps
        :param prediction: np.array of shape (B, W, H)
        :param target: np.array of shape (B, W, H)
        """
        b = len(timestamp)
        prediction = prediction.reshape(b, -1)
        target = target.reshape(b, -1)

        categorical_prediction = np.digitize(prediction, bins=self.evaluation_thresholds)
        categorical_target = np.digitize(target, bins=self.evaluation_thresholds)

        # load regional masks into shape of (19, -1)
        masks = np.random.randint(2, size=(19, 83, 67)) # TODO GET_MASK_HERE # (M, W, H) # TODO -1 for missing value
        masks = masks.reshape(19, -1)

        # split the prediction and target into 19 regions
        categorical_prediction = np.repeat(categorical_prediction[:, np.newaxis, :], 19, axis=1) # (B, R, W*H)
        categorical_target = categorical_target[:, np.newaxis, :] * masks[np.newaxis, :, :] # (B, R, W*H) 

        # calculate confusion matrix    
        classification_output = calculate_confusion_matrix(categorical_prediction.reshape(b*19, -1), categorical_target.reshape(b*19, -1),
                                                           n_category=len(self.evaluation_thresholds)+1, device=self.device) # (B*R, 4*4)

        # save into dataframe
        dates = [t.origin for t in timestamp for _ in range(19)]
        lead_times = [t.lead_time for t in timestamp for _ in range(19)]
        regions = list(range(19)) * b

        classification_output = classification_output.reshape(b, 19, 4, 4)

        new_data = pd.DataFrame({
                'date': dates,
                'leadtime': lead_times,
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
            })
        self.grid_df = pd.concat([self.grid_df, new_data], ignore_index=True)

    def save_df(self, df: pd.DataFrame, path: str) -> None:
        """
        Save dataframe to path
        """
        df.to_csv(path, index=False)

    def print_raw_data(self, raw_data: pd.DataFrame):
        """
        Print regression and classification metrics for raw data (self.regional_df)
        :param raw_data: pd.DataFrame;  self.regional_df
        """
        # regression
        target = np.array(raw_data['target_cont'])
        predictions = np.array(raw_data['predict_cont'])
        
        nmb = np.sum(predictions - target) / np.sum(target)
        nme = np.sum(np.abs(predictions - target)) / np.sum(np.abs(target))
        r, _ = pearsonr(predictions, target)
        rmse = np.sqrt(mean_squared_error(target, predictions))

        print(f"nmb: {round(nmb, 2)}, nme: {round(nme, 2)}, r: {round(r, 2)}, rmse: {round(rmse, 2)}")

        # classification
        predictions_cate = np.array(raw_data['predict_cate']).reshape(-1, 1).astype(int) # reshape(1, -1)
        target_cate = np.array(raw_data['target_cate']).reshape(-1, 1).astype(int) # reshape(1, -1)
        confusion_matrix = calculate_confusion_matrix(predictions_cate, target_cate, n_category=len(self.evaluation_thresholds)+1, device=self.device)
        
        self.print_confusion_matrix(confusion_matrix)

    def print_confusion_matrix(self, cm: np.array, bound:int=2):
        """
        Print classification metrics for confusion matrix (self.grid_df and self.regional_df)
        :param cm: np.array of shape (B, n_category, n_category)
        """
        # classification
        g01p01 = np.sum(cm[:, :bound, :bound])
        g01p23 = np.sum(cm[:, :bound, bound:])
        g23p01 = np.sum(cm[:, bound:, :bound])
        g23p23 = np.sum(cm[:, bound:, bound:])
        total = np.sum(cm)

        acc = np.sum(cm[:, 0, 0] + cm[:, 1, 1] + cm[:, 2, 2] + cm[:, 3, 3]) / total
        hard_acc = np.sum(cm[:, 2, 2] + cm[:, 3, 3]) / (g23p01 + g23p23)
        far = g01p23 / (g01p23 + g23p23)
        pod = g23p23 / (g23p23 + g23p01)
        f1 = 2 * pod * (1 - far) / (pod + (1 - far))

        print(f"acc: {round(acc, 2)}, hard_acc: {round(hard_acc, 2)}, far: {round(far, 2)}, pod: {round(pod, 2)}, f1: {round(f1, 2)}")