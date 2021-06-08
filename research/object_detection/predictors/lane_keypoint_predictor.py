"""Mask R-CNN Keypoint Predictor."""
import tensorflow as tf

from object_detection.core import keypoint_predictor
KEYPOINT_PREDICTIONS = keypoint_predictor.KEYPOINT_PREDICTIONS

class LaneKeypointPredictor(keypoint_predictor.KeypointPredictor):
    def __init__(self,
                 is_training,
                 num_keypoints,
                 keypoint_prediction_head):
        """Constructor.

        Args:
          is_training: Indicates whether the BoxPredictor is in training mode.
          num_keypoints: number of keypoints.
          keypoint_prediction_head: The head that predicts the keypoints.
        """
        super(LaneKeypointPredictor, self).__init__(is_training, num_keypoints)
        self._keypoint_prediction_head = keypoint_prediction_head

    @property
    def num_keypoints(self):
        return self._num_keypoints

    def _predict(self, image_features):
        if len(image_features) != 1:
            raise ValueError('length of `image_features` must be 1. Found {}'.format(
                len(image_features)))

        image_feature = image_features[0]
        predictions_dict = {}

        predictions_dict[KEYPOINT_PREDICTIONS] = self._keypoint_prediction_head.predict(
            roi_pooled_features=image_feature)

        return predictions_dict
