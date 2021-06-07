"""Function to build keypoint predictor from configuration."""

from object_detection.predictors import lane_keypoint_predictor
from object_detection.predictors.heads.lane_keypoint_head import LaneKeypointHead


def build(argscope_fn, keypoint_predictor_config, is_training, num_keypoints):
    """Builds keypoint predictor based on the configuration.
    """
    conv_hyperparams_fn = argscope_fn(keypoint_predictor_config.conv_hyperparams, is_training)

    keypoint_head = LaneKeypointHead(
        num_keypoints=num_keypoints,
        conv_hyperparams_fn=conv_hyperparams_fn,
        keypoint_heatmap_height=keypoint_predictor_config.keypoint_heatmap_height,
        keypoint_heatmap_width=keypoint_predictor_config.keypoint_heatmap_width,
        keypoint_prediction_num_conv_layers=keypoint_predictor_config.keypoint_prediction_num_conv_layers,
        keypoint_prediction_conv_depth=keypoint_predictor_config.keypoint_prediction_conv_depth
    )

    keypoint_predictor_object = lane_keypoint_predictor.LaneKeypointPredictor(
        is_training=is_training,
        num_keypoints=num_keypoints,
        keypoint_prediction_head=keypoint_head)

    return keypoint_predictor_object
