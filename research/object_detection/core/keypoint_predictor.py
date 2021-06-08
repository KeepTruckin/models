"""Keypoint predictor.

Keypoint predictors are classes that take a high level image features
as input and produce one prediction
1) a tensor encoding keyopint locations

This component is passed directly to loss functions
in our detection models.

These modules are separated from the main model since the same
few keypoint predictor architectures can be shared across many models.
"""

from abc import abstractmethod
import tensorflow as tf

KEYPOINT_PREDICTIONS = 'lane_keypoint_predictions'


class KeypointPredictor(object):
    """KeypointPredictor."""

    def __init__(self, is_training, num_keypoints):
        """Constructor.

        Args:
          is_training: Indicates whether the KeypointPredictor is in training mode.
          num_keypoints: number of keypoints.
        """
        self._is_training = is_training
        self._num_keypoints = num_keypoints

    @property
    def is_keras_model(self):
        return False

    @property
    def num_keypoints(self):
        return self._num_keypoints


    def predict(self, image_features, scope=None, **params):
        """Computes encoded keypoint locations.

            Takes a list of high level image feature maps as input and produces a list
            of keypoint encodings where each element in the output
            lists correspond to the feature maps in the input list.
        """
        if scope is not None:
            with tf.variable_scope(scope):
                return self._predict(image_features)
        return self._predict(image_features)

    @abstractmethod
    def _predict(self, image_features):
        """Implementations must override this method.
        """
        pass


class KerasKeypointPredictor(tf.keras.Model):
  """Keras-based KeypointPredictor."""

  def __init__(self, is_training, num_keyoints, freeze_batchnorm,
               inplace_batchnorm_update):
    """Constructor.

    Args:
      is_training: Indicates whether the KeypointPredictor is in training mode.
      num_keypoints: number of keypoints.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
    """
    super(KerasKeypointPredictor, self).__init__()

    self._is_training = is_training
    self._num_keyoints = num_keyoints
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

  @property
  def is_keras_model(self):
    return True

  @property
  def num_keyoints(self):
    return self._num_keyoints

  def call(self, image_features, scope=None, **kwargs):
    return self._predict(image_features)

  @abstractmethod
  def _predict(self, image_features, **kwargs):
      """Implementations must override this method.
      """
      raise NotImplementedError
