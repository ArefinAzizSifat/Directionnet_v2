import os
import collections

import tensorflow as tf
import util

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '', 'Directory where the data is stored')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_boolean('training', True, 'Mode')
flags.DEFINE_boolean('load_estimated_rot', False,
                     'Whether to load estimated rotations from disk.')


def data_loader(
        data_path,
        epochs,
        batch_size,
        training=True,
        load_estimated_rot=False):
    """
    Load stereo image datasets (TF2 version).

    Args:
      data_path: (string)
      epochs: (int) the number of training epochs.
      batch_size: (int) batch size.
      training: (bool) set it True when training to enable illumination
        randomization for input images.
      load_estimated_rot: (bool) set it True when training DirectionNet-T to
        load estimated rotation from DirectionNet-R saved as 'rotation_pred'
        on disk.

    Returns:
      tf.data.Dataset yielding a namedtuple:
      (id, src_image, trt_image, rotation, translation, fov, rotation_pred)
    """

    # ---------- Inner helpers -------------------------------------------------
    def load_data(path):
        """Given a directory path tensor, read its pickle files via numpy_function.

        path: tf.string scalar, e.g. b'/.../sample_00001'
        """

        # Build full pickle paths using tf.strings.join (TF2-safe)
        rot_path = tf.strings.join([path, '/rotation_gt.pickle'])
        epi_path = tf.strings.join([path, '/epipoles_gt.pickle'])
        fov_path = tf.strings.join([path, '/fov.pickle'])

        img_id, rotation = tf.numpy_function(
            util.read_pickle,
            [rot_path],
            [tf.string, tf.float32]
        )

        _, translation = tf.numpy_function(
            util.read_pickle,
            [epi_path],
            [tf.string, tf.float32]
        )

        _, fov = tf.numpy_function(
            util.read_pickle,
            [fov_path],
            [tf.string, tf.float32]
        )

        if load_estimated_rot:
            rot_pred_path = tf.strings.join([path, '/rotation_pred.pickle'])
            _, rotation_pred = tf.numpy_function(
                util.read_pickle,
                [rot_pred_path],
                [tf.string, tf.float32]
            )
        else:
            rotation_pred = tf.zeros_like(rotation)

        img_path = tf.strings.join([path, '/', img_id])
        return tf.data.Dataset.from_tensor_slices(
            (img_id, img_path, rotation, translation, fov, rotation_pred)
        )

    def load_images(img_id, img_path, rotation, translation, fov, rotation_pred):
        """Load images and decode text lines."""

        def load_single_image(one_img_path):
            image_bytes = tf.io.read_file(one_img_path)
            image = tf.image.decode_png(image_bytes, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.reshape(image, [512, 512, 3])
            image = tf.image.resize(
                image,
                [256, 256],
                method=tf.image.ResizeMethod.AREA
            )
            return image

        input_pair = collections.namedtuple(
            'data_input',
            [
                'id',
                'src_image',
                'trt_image',
                'rotation',
                'translation',
                'fov',
                'rotation_pred'
            ]
        )

        # Build full filenames with tf.strings.join
        src_path = tf.strings.join([img_path, '.src.perspective.png'])
        trt_path = tf.strings.join([img_path, '.trt.perspective.png'])

        src_image = load_single_image(src_path)
        trt_image = load_single_image(trt_path)

        if training:
            random_gamma = tf.random.uniform([], 0.7, 1.2)
            src_image = tf.image.adjust_gamma(src_image, random_gamma)
            trt_image = tf.image.adjust_gamma(trt_image, random_gamma)

        rotation = tf.reshape(rotation, [3, 3])
        rotation.set_shape([3, 3])

        translation = tf.reshape(translation, [3])
        translation.set_shape([3])

        fov = tf.reshape(fov, [1])
        fov.set_shape([1])

        if load_estimated_rot:
            rotation_pred = tf.reshape(rotation_pred, [3, 3])
            rotation_pred.set_shape([3, 3])

        return input_pair(
            img_id,
            src_image,
            trt_image,
            rotation,
            translation,
            fov,
            rotation_pred
        )

    ds = tf.data.Dataset.list_files(os.path.join(data_path, '*'))
    ds = ds.flat_map(load_data)

    ds = ds.map(
        load_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.repeat(epochs)
    # Keep final partial batches during evaluation to avoid silent drops that
    # would raise OutOfRangeError before metrics update.
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def main(argv=None):
    dataloader = data_loader(
        FLAGS.data_path,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.training,
        FLAGS.load_estimated_rot
    )

    for batch in dataloader.take(1):
        img_id, src_image, trt_image, rotation, translation, fov, rotation_pred = batch
        print("Batch details:")
        print("  Image ID:", img_id.numpy())
        print("  Source Image shape:", src_image.shape)
        print("  Target Image shape:", trt_image.shape)
        print("  Rotation shape:", rotation.shape)
        print("  Translation shape:", translation.shape)
        print("  FOV shape:", fov.shape)
        print("  Rotation_pred shape:", rotation_pred.shape)

    print("Dataloader executed successfully (TF2).")


if __name__ == '__main__':
    app.run(main)
