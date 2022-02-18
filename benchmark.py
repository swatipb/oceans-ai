import os
import time

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'Model path')
flags.DEFINE_string('image_path', None, 'Image path')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_string(
  'model_signature', 'serving_default', 'Signature of the model to run.')

_DEFAULT_SIGNATURE = 'serving_default'
_IMAGE_TYPE = tf.uint8
_NUM_ITER = 100

def parse_image(filename):
  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize(image, [1080, 1920], preserve_aspect_ratio=True)
  if _IMAGE_TYPE == tf.float32:
    image = tf.image.convert_image_dtype(image, tf.float32)
  else:
    image = tf.cast(tf.round(image), _IMAGE_TYPE)
  return filename, image


def main(unused_argv):
  list_ds = tf.data.Dataset.list_files(os.path.join(FLAGS.image_path, '*.jpg'))
  images_ds = list_ds.map(
    parse_image,
    num_parallel_calls=tf.data.AUTOTUNE
  ).prefetch(tf.data.AUTOTUNE)

  start = time.time()
  model = tf.saved_model.load(
    FLAGS.model_path,
    options=tf.saved_model.LoadOptions(allow_partial_checkpoint=True))

  try:
    serving_fn = model.signatures[FLAGS.model_signature]
  except KeyError:
    raise KeyError(
      f'Model does not have signautre {FLAGS.model_signature}. '
      f'Available signatures: {list(model.signatures)}')

  @tf.function(input_signature=[
    tf.TensorSpec((None, None, None, 3), _IMAGE_TYPE)])
  def model_fn(data):
    return serving_fn(data)

  print(f'model loaded in {time.time() - start:.3f}s')

  times = []
  num_images = 0
  
  # JIT warmup.
  num_tries = 0
  for filenames, images in images_ds.batch(FLAGS.batch_size).take(_NUM_ITER):
    num_tries += 1
    if num_tries >= 5:
      break
    _ = model_fn(images)

  for filenames, images in images_ds.batch(FLAGS.batch_size).take(_NUM_ITER):
    start = time.time()
    _ = model_fn(images)
    end = time.time()
    print(f'Inference in {end - start:.3f}s')
    times.append([end - start])
    num_images += len(filenames)
  print(f'Average time taken: {np.mean(times):.3f}s, std: {np.std(times):.3f}')
  print(f'Average FPS: {num_images / np.sum(times):.2f}')


if __name__ == '__main__':
    app.run(main)
