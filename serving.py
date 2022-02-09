"""Inference server. Currently works with TF OD API."""
from concurrent import futures
import logging

import grpc
import numpy as np
import service_pb2
import service_pb2_grpc
import tensorflow as tf

_MODEL_DIR = 'cots_efficientdet_d0/output/saved_model'
# _MODEL_DIR = 'cots_mobilenet_v2_ssd/saved_model'
_OUTPUT_NAMES = (
    'detection_anchor_indices',
    'detection_boxes',
    'detection_classes',
    'detection_multiclass_scores',
    'detection_scores',
    'num_detections',
    'raw_detection_boxes',
    'raw_detection_scores',
)
_DETECTION_THRESHOLD = 0.05
_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MiB
_IMAGE_TYPE = tf.uint8


class Detector(service_pb2_grpc.Detector):
  """Detector service. gets serizliaed tensors and returns detecion results."""

  def __init__(self, model):
    super().__init__()
    self._model = model

  def Inference(self, request, context):
    images = tf.io.parse_tensor(request.data, _IMAGE_TYPE)
    detections = self._model(images)
    result = service_pb2.InferenceReply()
    img_w, img_h = images.shape[1:3]

    num_detections = detections['num_detections'].numpy().astype(np.int32)
    detection_boxes = detections['detection_boxes'].numpy()
    detection_classes = detections['detection_classes'].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()

    print('Detected:', num_detections)
    for file_idx, file_path in enumerate(request.file_paths):
      scores = detection_scores[file_idx]
      print(scores)
      valid_indices = detection_scores[file_idx, :] >= _DETECTION_THRESHOLD
      scores = scores[valid_indices]
      classes = detection_classes[file_idx, valid_indices]
      bbox = detection_boxes[file_idx, valid_indices, :]

      for i, pos in enumerate(bbox):
        detection = service_pb2.BoundingBox(
            file_path=file_path,
            class_id=classes[i],
            score=scores[i],
            left=pos[0] * img_w,
            top=pos[1] * img_h,
            width=(pos[2] - pos[0]) * img_w,
            height=(pos[3] - pos[1]) * img_h,
        )
        result.detections.append(detection)
    return result


def serve():
  """Starts gRPC service."""
  model = tf.saved_model.load(_MODEL_DIR)
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=1),
      options=[
          ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
          ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
      ])
  service_pb2_grpc.add_DetectorServicer_to_server(Detector(model), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()


if __name__ == '__main__':
  logging.basicConfig()
  serve()
