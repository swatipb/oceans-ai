"""Library to track COTS using detected bounding boxes."""

import abc
import numpy as np
import service_pb2

from typing import Any, Dict
from collections import defaultdict
from object_detection.utils import np_box_ops

class Detection(abc.ABC):
  def __init__(self, bounding_box, file_name, is_detected=True):
    self.bounding_box = bounding_box
    self.is_detected = is_detected
    self.file_name = file_name

class RecentFrames(abc.ABC):
  def __init__(self, max_frames):
    self.frames = []
    self.max_frames = max_frames

  def insert_frame(self, frame_id):
    if (len(self.frames)) > self.max_frames:
      # Remove the oldest entry.
      self.frames.pop(0)
    self.frames.append(frame_id)

  def get_frames(self):
      return self.frames

  def get_max_frames(self):
    return self.max_frames

class CotsTracker(abc.ABC):
  def __init__(self):
    # See get_tracked_persons() for the format of the contents of this
    # dictionary.
    self.sequence_id_to_bbox = defaultdict(lambda: [])
    self.sequence_id = 0
    # Map from sequence id to number of objects linked to the sequence.
    self.sequence_id_to_length = defaultdict(int)
    self.min_iou = 0.4
    # Keep track of last n processed frames.
    self.recent_frames = RecentFrames(10)

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
    for sequence_id, sequence_list in self.sequence_id_to_bbox.items():
      previous_bbox = sequence_list[-1].bounding_box
      left = np.reshape(previous_bbox, (1, 4))
      right = np.reshape(bounding_box, (1, 4))
      if self.get_iou(left, right) > self.min_iou:
        return sequence_id
    return -1

  def remove_obsolete_sequences(self):
    if (len(self.recent_frames.get_frames())
        < self.recent_frames.get_max_frames()):
      # Not enough frames have been processed.
      return
    recent_frames = set(self.recent_frames.get_frames())
    sequence_ids_to_delete = []
    for sequence_id, sequence_list in self.sequence_id_to_bbox.items():
      recent_frame_matched_to_sequence = False
      # Process detections in reverse order - most recent to oldest
      for entry in reversed(sequence_list):
        if entry.file_name in recent_frames:
          recent_frame_matched_to_sequence = True
          break
      if not recent_frame_matched_to_sequence:
        sequence_ids_to_delete.append(sequence_id)
    for sequence_id in sequence_ids_to_delete:
      print('deleting sequence id', sequence_id)
      del self.sequence_id_to_bbox[sequence_id]
      del self.sequence_id_to_length[sequence_id]

  def process_frame(self, filename, detections):
    self.remove_obsolete_sequences()
    self.recent_frames.insert_frame(filename)
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
      bounding_box = np.array([detection.top, detection.left,
                               detection.top + detection.height,
                               detection.left + detection.width])
      current_detection = Detection(bounding_box, filename, True)
      detection_sequence_id = self._track_object(bounding_box)
      if detection_sequence_id == -1:
        # Unable to link detection to existing sequence.
        # Generate new sequence id
        detection_sequence_id = self._generate_next_sequence_id()
      file_sequence_id_to_bbox[detection_sequence_id].append(current_detection)
      tracker_result.sequence_id = detection_sequence_id
      self.sequence_id_to_length[detection_sequence_id] += 1
      tracker_result.sequence_length = (
          self.sequence_id_to_length[detection_sequence_id])

    # Merge detections from this file into class map.
    for key, value in file_sequence_id_to_bbox.items():
      self.sequence_id_to_bbox[key].extend(value)
    return results
