# """Anomaly metrics."""
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt


# def compute_imagewise_retrieval_metrics(
#     anomaly_prediction_weights, anomaly_ground_truth_labels, draw_roc_curve=False
# ):

#     """
#     Computes retrieval statistics (AUROC, FPR, TPR).

#     Args:
#         anomaly_prediction_weights: [np.array or list] [N] Assignment weights
#                                     per image. Higher indicates higher
#                                     probability of being an anomaly.
#         anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
#                                     if image is an anomaly, 0 if not.
#     """
#     fpr, tpr, thresholds = metrics.roc_curve(
#         anomaly_ground_truth_labels, anomaly_prediction_weights
#     )
#     auroc = metrics.roc_auc_score(
#         anomaly_ground_truth_labels, anomaly_prediction_weights
#     )
#     if draw_roc_curve:
#         draw_curve(fpr, tpr, auroc)

    
#     return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


# def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, draw_roc_curve=False):
#     """
#     Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
#     and ground truth segmentation masks.

#     Args:
#         anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
#                                 generated segmentation masks.
#         ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
#                             predefined ground truth segmentation masks
#     """
#     if isinstance(anomaly_segmentations, list):
#         anomaly_segmentations = np.stack(anomaly_segmentations)
#     if isinstance(ground_truth_masks, list):
#         ground_truth_masks = np.stack(ground_truth_masks)

#     flat_anomaly_segmentations = anomaly_segmentations.ravel()
#     flat_ground_truth_masks = ground_truth_masks.ravel()

#     fpr, tpr, thresholds = metrics.roc_curve(
#         flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
#     )
#     auroc = metrics.roc_auc_score(
#         flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
#     )

#     precision, recall, thresholds = metrics.precision_recall_curve(
#         flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
#     )
#     f1_scores = np.divide(
#         2 * precision * recall,
#         precision + recall,
#         out=np.zeros_like(precision),
#         where=(precision + recall) != 0,
#     )

#     optimal_threshold = thresholds[np.argmax(f1_scores)]
#     predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
#     fpr_optim = np.mean(predictions > flat_ground_truth_masks)
#     fnr_optim = np.mean(predictions < flat_ground_truth_masks)

#     if draw_roc_curve:
#         draw_curve(fpr, tpr, auroc)

#     return {
#         "auroc": auroc,
#         "fpr": fpr,
#         "tpr": tpr,
#         "optimal_threshold": optimal_threshold,
#         "optimal_fpr": fpr_optim,
#         "optimal_fnr": fnr_optim,
#     }


# def draw_curve(fpr, tpr, auroc):
#     plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.4f})'.format(auroc), lw=2)

#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc="lower right")

#     error = 0.015
#     miss = 0.1
#     plt.plot([error, error], [-0.05, 1.05], 'k:', lw=1)
#     plt.plot([-0.05, 1.05], [1-miss, 1-miss], 'k:', lw=1)
#     error_y, miss_x = 0, 1
#     for i in range(len(fpr)):
#         if fpr[i] <= error <= fpr[i + 1]:
#             error_y = tpr[i]
#         if tpr[i] <= 1-miss <= tpr[i + 1]:
#             miss_x = fpr[i]
#     # plt.scatter(error, error_y, c='k')
#     # plt.scatter(miss_x, 1-miss, c='k')
#     plt.text(error, error_y, "({0}, {1:.4f})".format(error, error_y), color='k')
#     plt.text(miss_x, 1-miss, "({0:.4f}, {1})".format(miss_x, 1-miss), color='k')
#     plt.savefig("roc_curve.png")
#     plt.show()


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def compute_dice_score(predictions, targets, threshold=0.5):
    """
    Compute Dice Score.

    Args:
        predictions (np.ndarray or list): Predicted segmentation masks.
        targets (np.ndarray or list): Ground truth segmentation masks.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        float: Dice Score.
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(targets, list):
        targets = np.array(targets)
        
    predictions = (predictions > threshold).astype(int)
    targets = (targets > threshold).astype(int)
    intersection = np.sum(predictions * targets)
    dice_score = (2 * intersection) / (np.sum(predictions) + np.sum(targets))
    return dice_score


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, draw_roc_curve=False
):

    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    if draw_roc_curve:
        draw_curve(fpr, tpr, auroc)

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, draw_roc_curve=False):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(f1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    if draw_roc_curve:
        draw_curve(fpr, tpr, auroc)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def draw_curve(fpr, tpr, auroc):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.4f})'.format(auroc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    error = 0.015
    miss = 0.1
    plt.plot([error, error], [-0.05, 1.05], 'k:', lw=1)
    plt.plot([-0.05, 1.05], [1-miss, 1-miss], 'k:', lw=1)
    error_y, miss_x = 0, 1
    for i in range(len(fpr)):
        if fpr[i] <= error <= fpr[i + 1]:
            error_y = tpr[i]
        if tpr[i] <= 1-miss <= tpr[i + 1]:
            miss_x = fpr[i]
    plt.text(error, error_y, "({0}, {1:.4f})".format(error, error_y), color='k')
    plt.text(miss_x, 1-miss, "({0:.4f}, {1})".format(miss_x, 1-miss), color='k')
    plt.savefig("roc_curve.png")
    plt.show()


def plot_dice_scores_vs_noise(dice_scores_collect, save_path):
    """
    Plots Dice scores for different noise levels.

    Args:
        dice_scores_collect (list): A list of dictionaries containing dataset names, noise levels, and dice scores.
        save_path (str): Path to save the plot.
    """
    plt.figure()
    noise_levels = sorted(set(entry['noise_level'] for entry in dice_scores_collect))
    for dataset_name in set(entry['dataset_name'] for entry in dice_scores_collect):
        scores = [entry['dice_score'] for entry in dice_scores_collect if entry['dataset_name'] == dataset_name]
        plt.plot(noise_levels, scores, label=dataset_name)

    plt.xlabel("Noise Level")
    plt.ylabel("Dice Score")
    plt.title("Dice Score vs. Noise Level")
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice_scores_vs_noise.png"))
    plt.show()
