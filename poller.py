"""Watches specific directory and sends new file to detector service.

Usage:
  python poller.py path/to/watch
"""
import logging
import multiprocessing
import sys
import time
import cots_tracker
import os

from google.protobuf import text_format
import grpc
import service_pb2
import service_pb2_grpc
import tensorflow as tf
from watchdog import events
from watchdog import observers

_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MiB
_IMAGE_TYPE = tf.uint8

# https://hub.docker.com/r/helmuthva/jetson-xavier-tensorflow-serving
# https://velog.io/@canlion/tensorflow-2.x-od-api-tensorRT
# TensorFlow & lower perf VS. PyTorch & high perf...

rx, tx = multiprocessing.Pipe(duplex=False)


def data_gen():
  if rx.poll():
    yield rx.recv_bytes()


def parse_image(filename):
  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize(image, [1280, 720], preserve_aspect_ratio=True)
  if _IMAGE_TYPE == tf.float32:
    image = tf.image.convert_image_dtype(image, tf.float32)
  else:
    image = tf.cast(tf.round(image), _IMAGE_TYPE)
  return filename, image


def get_ordered_filename_to_detections(inference_response):
  result = defaultdict(lambda: [])
  for detection in inference_response.detections:
    head, tail = os.path.split(detection.file_path)
    result[tail].append(detection)
  return OrderedDict(sorted(result.items()))


def format_tracker_response(tracker_results):
  result = tracker_results.file_path
  for entry in tracker_results.tracker_results:
    detection_columns = [str(entry.detection.class_id),
                         str(entry.detection.score),
                         str(entry.sequence_id),
                         str(entry.sequence_length),
                         str(entry.detection.top),
                         str(entry.detection.left),
                         str(entry.detection.width),
                         str(entry.detection.height)]
    result += ", { " + ",".join(detection_columns) + "}"
  result += ","
  return result


class Handler(events.FileSystemEventHandler):

  def on_created(self, event):
    if event.is_directory:
      return
    print('detected', event.src_path)
    if event.src_path[-4:] != '.jpg':
      return
    tx.send_bytes(bytes(event.src_path, encoding='utf-8'))


if __name__ == '__main__':
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S')
  if (len(sys.argv) < 2):
    print('Output csv filepath not passed')
    sys.exit()

  output_filepath = sys.argv[1]
  path = sys.argv[2] if len(sys.argv) > 2 else '.'
  event_handler = Handler()
  observer = observers.Observer()
  observer.schedule(event_handler, path, recursive=True)
  observer.start()
  ds_counter = tf.data.Dataset.from_generator(
      data_gen,
      output_types=tf.string,
      output_shapes=(),
  )
  image_ds = ds_counter.map(parse_image)

  with grpc.insecure_channel(
      'localhost:50051',
      options=[
          ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
          ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
      ]) as channel:
    stub = service_pb2_grpc.DetectorStub(channel)
    tracker = cots_tracker.CotsTracker()
    try:
      while True:
        time.sleep(1)
        for data in image_ds.repeat().batch(1).take(1):
          try:
            response = stub.Inference(
                service_pb2.InferenceRequest(
                    file_paths=list(data[0].numpy()),
                    data=tf.io.serialize_tensor(data[1]).numpy()))
            print(type(response), text_format.MessageToString(response))
            filename_to_detections = get_ordered_filename_to_detections(response)
            try:
              output_file = open(output_filepath, 'a')
              for filename, detections in filename_to_detections.items():
                if not detections:
                  output_file.write(filename + ',')
                else:
                  tracker_results = tracker.process_frame(
                      filename, detections)
                  result_as_string = format_tracker_response(tracker_results)
                  output_file.write(result_as_string + '\n')
              output_file.close()
            except (OSError, IOError) as e:
              print('Error writing to file {0}'.format(e.strerror))            
          except grpc.RpcError as e:
            print(e)

    except KeyboardInterrupt:
      observer.stop()
    observer.join()
