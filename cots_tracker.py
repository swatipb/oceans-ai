"""Library to track COTS using detected bounding boxes."""

import abc
import numpy as np
import service_pb2

from typing import Any, Dict
from collections import defaultdict
from object_detection.utils import np_box_ops


class CotsTracker(abc.ABC):
  def __init__(self):
    # See get_tracked_persons() for the format of the contents of this
    # dictionary.
    self.sequence_id_to_bbox = defaultdict(lambda: [])
    self.sequence_id = 0
    # Map from sequence id to number of objects linked to the sequence.
    self.sequence_id_to_length = defaultdict(int)
    self.min_iou = 0.55

  def _increment_sequence_length(self, sequence_id):
    self.sequence_id_to_length[sequence_id] += 1

  def _generate_next_sequence_id(self):
  # Returns an ID for newly tracked person."""
    sequence_id = self.sequence_id
    self.sequence_id += 1
    return sequence_id

  def get_iou(self, ground_truth, prediction):
    return np_box_ops.iou(ground_truth, prediction)

  def _track_object(self, bounding_box):
    # Returns existing sequence id if object is tracked else returns -1
    for sequence_id, bbox_list in self.sequence_id_to_bbox.items():
      print('sequence_id', sequence_id)
      print('bbox_list', np.reshape(bbox_list[-1], (1, 4)))
      print('bounding_box', np.reshape(bounding_box, (1, 4)))
      left = np.reshape(bbox_list[-1], (1, 4))
      right = np.reshape(bounding_box, (1, 4))
      if self.get_iou(left, right) > self.min_iou:
        return sequence_id
    return -1

  def process_frame(self, filename, detections):
    results = service_pb2.TrackerResults()
    results.file_path = filename

    # Store mapping from sequence id to bounding boxes tracked in a single
    # image. Assumption: Two detections within the same file cannot belong to
    # the same sequence. Merge sequences detected in a file after tracking for
    # objects detected in a image is complete.
    file_sequence_id_to_bbox = defaultdict(lambda: [])

    for detection in detections:
      tracker_result = results.tracker_results.add()
      tracker_result.detection.CopyFrom(detection)
      file_path = detection.file_path
      bounding_box = np.array([detection.top, detection.left,
                               detection.top + detection.height,
                               detection.left + detection.width])
      detection_sequence_id = self._track_object(bounding_box)
      if detection_sequence_id == -1:
        # Unable to link detection to existing sequence.
        # Generate new sequence id
        detection_sequence_id = self._generate_next_sequence_id()
      file_sequence_id_to_bbox[detection_sequence_id].append(bounding_box)
      tracker_result.sequence_id = detection_sequence_id
      self.sequence_id_to_length[detection_sequence_id] += 1
      tracker_result.sequence_length = (
          self.sequence_id_to_length[detection_sequence_id])

    # Merge detections from this file into class map.
    for key,value in file_sequence_id_to_bbox.items():
      self.sequence_id_to_bbox[key].extend(value)
    self._increment_sequence_length(detection_sequence_id)
    return results
