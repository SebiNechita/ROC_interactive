import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve)
from distributions import NormalDistData


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Metrics:
    """Calculates initial and updated binary classification metrics based on 2 input distributions."""

    def __init__(self, dist0: NormalDistData, dist1: NormalDistData):
        # All interactive & inter-related, so pass the instances and keep as ColumnDataSources as much as possible
        self.dist0 = dist0
        self.dist1 = dist1
        self.max_dist_y_value = self._get_dist_plot_max()

        # "True" binary labels
        # assigns 0s to the first distribution, 1s to the 2nd distribution
        self.y_true = ColumnDataSource(
            data=dict(
                data=np.concatenate(
                    (
                        np.zeros_like(self.dist0.raw_data.data["data"]),
                        np.ones_like(self.dist1.raw_data.data["data"]),
                    )
                )
            )
        )

        # Actual values from distribution
        # to compare against self.y_true for constructing an ROC curve or other
        self.y_score = ColumnDataSource(
            data=dict(
                data=np.concatenate(
                    (self.dist0.raw_data.data["data"], self.dist1.raw_data.data["data"])
                )
            )
        )

        # Make ROC Curve data
        (
            self.roc_curve,
            self.roc_thresholds,
            self.roc_threshold_dot,
        ) = self._make_roc_curve()

        # Threshold line graphical object construction
        self.threshold_line = ColumnDataSource(
            data=dict(
                x=[
                    self.roc_thresholds.data["thresholds"].min(),
                    self.roc_thresholds.data["thresholds"].min(),
                ],
                y=[0.0, self._get_dist_plot_max(),],
            )
        )

        # Make confusion matrix
        self.cm = self._get_confusion_matrix()

        # Metrics like AUC, recall, f1, etc.
        self.metrics = self._get_metrics()

    def _get_dist_plot_max(self):
        max0 = self.dist0.kde_curve.data["y"].max()
        max1 = self.dist1.kde_curve.data["y"].max()
        max_dist_y_value = np.maximum(max0, max1)
        return max_dist_y_value

    def _update_y(self):
        """Update predicted and true lables (`y` in the machine learning context). Not to be confused with (x,y) axes in plots."""
        y_true = dict(
            data=np.concatenate(
                (
                    np.zeros_like(self.dist0.raw_data.data["data"]),
                    np.ones_like(self.dist1.raw_data.data["data"]),
                )
            )
        )
        y_score = dict(
            data=np.concatenate(
                (self.dist0.raw_data.data["data"], self.dist1.raw_data.data["data"])
            )
        )
        self.y_true.data = y_true
        self.y_score.data = y_score
        return self.y_true, self.y_score

    def _make_roc_curve(self):
        # magic number, change to threshold_slider.value later if dot_handler inadequate
        INITIAL_THRESHOLD_VALUE = 0.0
        # Calculate ROC curve coordinates
        fpr, tpr, thresh = roc_curve(
            self.y_true.data["data"], self.y_score.data["data"]
        )
        # ROC CDS includes data for shading area under curve: y_lower & upper
        curve = ColumnDataSource(
            data=dict(x=fpr, y_upper=tpr, y_lower=np.zeros_like(tpr))
        )
        thresholds = ColumnDataSource(data=dict(thresholds=thresh))
        # Find nearest ROC point based on threshold
        roc_curve_idx = find_nearest_idx(
            thresholds.data["thresholds"], INITIAL_THRESHOLD_VALUE
        )
        dot = ColumnDataSource(
            data=dict(x=[fpr[roc_curve_idx]], y=[tpr[roc_curve_idx]])
        )
        return curve, thresholds, dot

    def _get_confusion_matrix(self):
        y_true = self.y_true.data["data"]
        y_score = self.y_score.data["data"]
        # Calculate predictions based on threshold
        thresh = self.threshold_line.data["x"][0]
        y_pred = (y_score >= thresh).astype("int")
        # Create confusion matrix data structure for plotting
        cm = confusion_matrix(y_true, y_pred)
        df = pd.DataFrame(
            {
                "x": [0, 1, 0, 1],
                "y": [1, 1, 0, 0],
                "cm_values": np.flip(cm, axis=0).flatten(),
                "value_coord_x": [0, 1, 0, 1],
                "value_coord_y": [1, 1, 0, 0],
            }
        )
        cm = ColumnDataSource(df)
        return cm

    def _get_metrics(self):
        # Convert for readability within this function
        y_true = self.y_true.data["data"]
        y_score = self.y_score.data["data"]

        thresh = self.threshold_line.data["x"][0]
        y_pred = (y_score >= thresh).astype("int")

        # Get metrics
        auc = roc_auc_score(y_true, y_score)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = ColumnDataSource(data=dict(auc=[auc]))
        metrics.data["accuracy"] = [accuracy]
        return metrics

    # Threshold callback handlers (green line & dot)
    def threshold_line_x_handler(self, attr, old, new):
        """Controls horizontal movement when threshold changes."""
        self.threshold_line.data = dict(x=[new, new], y=[0.0, self.max_dist_y_value])

    def threshold_line_y_handler(self, attr, old, new):
        """Controls height changes when distributions change (dist.n or dist.sd)."""
        self.max_dist_y_value = self._get_dist_plot_max()
        x = self.threshold_line.data["x"][0]
        self.threshold_line.data = dict(x=[x, x], y=[0.0, self.max_dist_y_value])

    def roc_threshold_dot_handler(self, attr, old, new):
        """Update coordinates of roc_threshold dot."""
        roc_curve_idx = find_nearest_idx(self.roc_thresholds.data["thresholds"], new)
        fpr = self.roc_curve.data["x"]
        tpr = self.roc_curve.data["y_upper"]
        x, y = fpr[roc_curve_idx], tpr[roc_curve_idx]
        self.roc_threshold_dot.data = dict(x=[x], y=[y])

    def roc_curve_handler(self, attr, old, new):
        """Update ROC curve when distributions change."""
        self._update_y()
        new_curve, new_thresholds, new_dot = self._make_roc_curve()
        self.roc_curve.data = dict(new_curve.data)
        self.roc_thresholds.data = dict(new_thresholds.data)
        self.roc_threshold_dot.data = dict(new_dot.data)

    def metrics_handler(self, attr, old, new):
        """Update metric values when distributions change."""
        self._update_y()
        metrics = self._get_metrics()
        self.metrics.data = dict(metrics.data)

    def cm_handler(self, attr, old, new):
        """Update confusion marix values."""
        self._update_y()
        cm = self._get_confusion_matrix()
        self.cm.data = dict(cm.data)
