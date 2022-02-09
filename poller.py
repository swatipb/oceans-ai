"""Watches specific directory and sends new file to detector service.

Usage:
  python poller.py path/to/watch
"""
import logging
import multiprocessing
import sys
import time

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
  path = sys.argv[1] if len(sys.argv) > 1 else '.'
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
          except grpc.RpcError as e:
            print(e)

    except KeyboardInterrupt:
      observer.stop()
    observer.join()
