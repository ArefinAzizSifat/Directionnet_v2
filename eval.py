# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- TF1 compatibility + tf_slim patch block (for TF2) ---
import util
import tf_slim.metrics as metrics
import tensorflow_probability as tfp
import model
import dataset_loader
from absl import flags
from absl import app
from tensorflow.python.ops import array_ops as _aops
from tensorflow.python.ops import control_flow_ops as _cfo
import tensorflow as tf
import math
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Use TF1-style graph mode on TF2
tf.compat.v1.disable_v2_behavior()
tf1 = tf.compat.v1

# tf_slim sometimes expects these old symbols
if not hasattr(_cfo, "cond"):
    _cfo.cond = tf1.cond

if not hasattr(_aops, "stack"):
    _aops.stack = tf.stack
# --- end of patch block ---

"""Evaluate variations of DirectionNet for relative camera pose estimation."""

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'eval_data_dir', '',
    'The test data directory.')
flags.DEFINE_string('save_summary_dir', '', 'The directory to save summary.')
flags.DEFINE_string(
    'checkpoint_dir', '', 'The directory to load the checkpoints.')
flags.DEFINE_string(
    'model', '9D', '9D(rotation), T(translation), Single(DirectionNet-Single)')
flags.DEFINE_integer('batch', 1, 'Size of mini-batches.')
flags.DEFINE_integer(
    'distribution_height', 64, 'The height dimension of output distributions.')
flags.DEFINE_integer(
    'distribution_width', 64, 'The width dimension of output distributions.')
flags.DEFINE_integer(
    'transformed_height', 344,
    'The height dimension of input images after derotation transformation.')
flags.DEFINE_integer(
    'transformed_width', 344,
    'The width dimension of input images after derotation transformation.')
flags.DEFINE_float('kappa', 10.,
                   'A coefficient multiplied by the concentration loss.')
flags.DEFINE_float(
    'transformed_fov', 105.,
    'The field of view of input images after derotation transformation.')
flags.DEFINE_bool('derotate_both', True,
                  'Derotate both input images when training DirectionNet-T')
flags.DEFINE_integer('testset_size', 1000, 'The size of the test set.')
flags.DEFINE_integer(
    'eval_interval_secs', 5 * 60, 'Evaluation interval. default: 5 mins')


def streaming_median_metric(values):
    """Record streaming median for tf metrics."""
    values_concat, values_concat_op = metrics.streaming_concat(values)
    values_vec = tf1.reshape(values_concat, (-1,))
    count = tf1.size(values_vec)

    def compute_percentile():
        return tfp.stats.percentile(values_vec, 50)

    def empty_percentile():
        return tf1.constant(float('nan'), dtype=values_vec.dtype)

    median = tf1.cond(count > 0, compute_percentile, empty_percentile)
    return median, values_concat_op


def eval_direction_net_rotation(src_img,
                                trt_img,
                                rotation_gt,
                                n_output_distributions=3):
    """Evaluate the DirectionNet-R.

    Args:
      src_img: [BATCH, HEIGHT, WIDTH, 3] input source images.
      trt_img: [BATCH, HEIGHT, WIDTH, 3] input target images.
      rotation_gt: [BATCH, 3, 3] ground truth rotation matrices.
      n_output_distributions: (int) number of output distributions. (either two or
      three) The model uses 9D representation for rotations when it is 3 and the
      model uses 6D representation when it is 2.

    Returns:
      Tensorflow metrics.

    Raises:
      ValueError: 'n_output_distributions' must be either 2 or 3.
    """
    if n_output_distributions != 3 and n_output_distributions != 2:
        raise ValueError("'n_output_distributions' must be either 2 or 3.")

    net = model.DirectionNet(n_output_distributions)
    directions_gt = rotation_gt[:, :n_output_distributions]
    distribution_gt = util.spherical_normalization(util.von_mises_fisher(
        directions_gt,
        tf.constant(FLAGS.kappa, tf.float32),
        [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

    pred = net(src_img, trt_img, training=False)
    directions, _, distribution_pred = util.distributions_to_directions(pred)
    if n_output_distributions == 3:
        rotation_estimated = util.svd_orthogonalize(directions)
    elif n_output_distributions == 2:
        rotation_estimated = util.gram_schmidt(directions)
    angular_errors = util.angular_distance(directions, directions_gt)
    x_error = tf1.reduce_mean(angular_errors[:, 0])
    y_error = tf1.reduce_mean(angular_errors[:, 1])
    z_error = tf1.reduce_mean(angular_errors[:, 2])
    rotation_error = tf1.reduce_mean(util.rotation_geodesic(
        rotation_estimated, rotation_gt))

    for i in range(n_output_distributions):
        tf1.summary.image('distribution/rotation/ground_truth_%d' % (i + 1),
                          distribution_gt[:, :, :, i:i + 1],
                          max_outputs=4)
        tf1.summary.image('distribution/rotation/prediction_%d' % (i + 1),
                          distribution_pred[:, :, :, i:i + 1],
                          max_outputs=4)

    tf1.summary.image('source_image', src_img, max_outputs=4)
    tf1.summary.image('target_image', trt_img, max_outputs=4)

    metrics_to_values, metrics_to_updates = (
        metrics.aggregate_metric_map({
            'angular_error/x': tf1.metrics.mean(
                util.radians_to_degrees(x_error)),
            'angular_error/y': tf1.metrics.mean(
                util.radians_to_degrees(y_error)),
            'angular_error/z': tf1.metrics.mean(
                util.radians_to_degrees(z_error)),
            'rotation_error': tf1.metrics.mean(
                util.radians_to_degrees(rotation_error)),
            'rotation_error/median': streaming_median_metric(
                tf1.reshape(util.radians_to_degrees(rotation_error), (1,)))
        }))
    return metrics_to_values, metrics_to_updates


def eval_direction_net_translation(src_img,
                                   trt_img,
                                   rotation_gt,
                                   translation_gt,
                                   fov_gt,
                                   rotation_pred,
                                   derotate_both=False):
    """Evaluate the DirectionNet-T."""
    net = model.DirectionNet(1)

    (transformed_src, transformed_trt) = util.derotation(
        src_img,
        trt_img,
        rotation_pred,
        fov_gt,
        FLAGS.transformed_fov,
        [FLAGS.transformed_height, FLAGS.transformed_width],
        derotate_both)

    (transformed_src_gt, transformed_trt_gt) = util.derotation(
        src_img,
        trt_img,
        rotation_gt,
        fov_gt,
        FLAGS.transformed_fov,
        [FLAGS.transformed_height, FLAGS.transformed_width],
        derotate_both)

    translation_gt = tf1.expand_dims(translation_gt, 1)
    distribution_gt = util.spherical_normalization(util.von_mises_fisher(
        translation_gt,
        tf.constant(FLAGS.kappa, tf.float32),
        [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

    pred = net(transformed_src, transformed_trt, training=False)
    directions, _, distribution_pred = util.distributions_to_directions(pred)

    half_derotation = util.half_rotation(rotation_pred)
    # The output directions are relative to the derotated frame. Transform them
    # back to the source images' frame.
    directions = tf1.matmul(directions, half_derotation, transpose_b=True)
    translation_error = tf1.reduce_mean(tf1.acos(tf1.clip_by_value(
        tf1.reduce_sum(directions * translation_gt, -1), -1., 1.)))

    tf1.summary.image('distribution/translation/ground_truth',
                      distribution_gt,
                      max_outputs=4)
    tf1.summary.image('distribution/translation/prediction',
                      distribution_pred,
                      max_outputs=4)

    tf1.summary.image('source_image', src_img, max_outputs=4)
    tf1.summary.image('target_image', trt_img, max_outputs=4)
    tf1.summary.image('transformed_source_image',
                      transformed_src, max_outputs=4)
    tf1.summary.image('transformed_target_image',
                      transformed_trt, max_outputs=4)
    tf1.summary.image(
        'transformed_source_image_gt', transformed_src_gt, max_outputs=4)
    tf1.summary.image(
        'transformed_target_image_gt', transformed_trt_gt, max_outputs=4)

    metrics_to_values, metrics_to_updates = (
        metrics.aggregate_metric_map({
            'translation_error':
                tf1.metrics.mean(util.radians_to_degrees(translation_error)),
            'translation_error/median':
                streaming_median_metric(
                    tf1.reshape(
                        util.radians_to_degrees(translation_error), (1,)))
        }))
    return metrics_to_values, metrics_to_updates


def eval_direction_net_single(src_img,
                              trt_img,
                              rotation_gt,
                              translation_gt):
    """Evaluate the DirectionNet-Single."""
    net = model.DirectionNet(4)
    translation_gt = tf1.expand_dims(translation_gt, 1)
    directions_gt = tf1.concat([rotation_gt, translation_gt], 1)
    distribution_gt = util.spherical_normalization(util.von_mises_fisher(
        directions_gt,
        tf.constant(FLAGS.kappa, tf.float32),
        [FLAGS.distribution_height, FLAGS.distribution_width]), rectify=False)

    pred = net(src_img, trt_img, training=False)
    directions, _, distribution_pred = util.distributions_to_directions(pred)
    rotation_estimated = util.svd_orthogonalize(
        directions[:, :3])

    angular_errors = util.angular_distance(directions, directions_gt)
    x_error = tf1.reduce_mean(angular_errors[:, 0])
    y_error = tf1.reduce_mean(angular_errors[:, 1])
    z_error = tf1.reduce_mean(angular_errors[:, 2])
    translation_error = tf1.reduce_mean(angular_errors[:, 3])
    rotation_error = tf1.reduce_mean(util.rotation_geodesic(
        rotation_estimated, rotation_gt))

    for i in range(4):
        tf1.summary.image('distribution/rotation/ground_truth_%d' % (i + 1),
                          distribution_gt[:, :, :, i:i + 1],
                          max_outputs=4)
        tf1.summary.image('distribution/rotation/prediction_%d' % (i + 1),
                          distribution_pred[:, :, :, i:i + 1],
                          max_outputs=4)

    tf1.summary.image('distribution/translation/ground_truth',
                      distribution_gt[:, :, :, -1:],
                      max_outputs=4)
    tf1.summary.image('distribution/translation/prediction',
                      distribution_pred[:, :, :, -1:],
                      max_outputs=4)

    tf1.summary.image('source_image', src_img, max_outputs=4)
    tf1.summary.image('target_image', trt_img, max_outputs=4)

    metrics_to_values, metrics_to_updates = (
        metrics.aggregate_metric_map({
            'angular_error/x':
                tf1.metrics.mean(util.radians_to_degrees(x_error)),
            'angular_error/y':
                tf1.metrics.mean(util.radians_to_degrees(y_error)),
            'angular_error/z':
                tf1.metrics.mean(util.radians_to_degrees(z_error)),
            'rotation_error':
                tf1.metrics.mean(util.radians_to_degrees(rotation_error)),
            'rotation_error/median':
                streaming_median_metric(
                    tf1.reshape(
                        util.radians_to_degrees(rotation_error), (1,))),
            'translation_error':
                tf1.metrics.mean(util.radians_to_degrees(translation_error)),
            'translation_error/median':
                streaming_median_metric(
                    tf1.reshape(
                        util.radians_to_degrees(translation_error), (1,)))
        }))
    return metrics_to_values, metrics_to_updates


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Build dataset (one pass over eval set)
    ds = dataset_loader.data_loader(
        data_path=FLAGS.eval_data_dir,
        epochs=1,
        batch_size=FLAGS.batch,
        training=False,
        load_estimated_rot=(FLAGS.model == 'T')
    )
    if FLAGS.testset_size and FLAGS.testset_size > 0:
        # The loader already returns batched elements, so cap the number of
        # batches we will consume instead of overshooting and waiting for more
        # data than requested.
        num_batches = int(math.ceil(float(FLAGS.testset_size) / FLAGS.batch))
        if num_batches == 0:
            raise ValueError('testset_size must be >= batch size when > 0.')
        ds = ds.take(num_batches)
    elements = tf1.data.make_one_shot_iterator(ds).get_next()
    src_img, trt_img = elements.src_image, elements.trt_image
    rotation_gt = elements.rotation
    translation_gt = elements.translation

    # Build metrics graph depending on model type
    if FLAGS.model == '9D':
        metrics_to_values, metrics_to_updates = eval_direction_net_rotation(
            src_img, trt_img, rotation_gt, 3)
    elif FLAGS.model == '6D':
        metrics_to_values, metrics_to_updates = eval_direction_net_rotation(
            src_img, trt_img, rotation_gt, 2)
    elif FLAGS.model == 'T':
        fov_gt = tf1.squeeze(elements.fov, -1)
        rotation_pred = elements.rotation_pred
        metrics_to_values, metrics_to_updates = eval_direction_net_translation(
            src_img, trt_img, rotation_gt, translation_gt, fov_gt, rotation_pred,
            FLAGS.derotate_both)
    elif FLAGS.model == 'Single':
        metrics_to_values, metrics_to_updates = eval_direction_net_single(
            src_img, trt_img, rotation_gt, translation_gt)
    else:
        raise ValueError("Unknown model type: {}".format(FLAGS.model))

    # Add summaries for metrics (with tf.Print for logging)
    for name, value in metrics_to_values.items():
        tf1.summary.scalar('eval/' + name, tf1.Print(value, [value], name))

    # Group all metric update ops so we can control execution order.
    update_group = tf1.group(
        *metrics_to_updates.values(), name='metrics_update')

    # Merge all summaries
    summary_op = tf1.summary.merge_all()
    with tf1.control_dependencies([update_group]):
        summary_with_updates = tf1.identity(
            summary_op, name='summary_with_updates')

    # Saver to restore variables from checkpoint
    saver = tf1.train.Saver()

    with tf1.Session() as sess:
        # Init variables (will be overwritten by restore)
        sess.run(tf1.global_variables_initializer())
        sess.run(tf1.local_variables_initializer())

        # Restore latest checkpoint
        ckpt_path = tf1.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt_path is None:
            raise RuntimeError(
                "No checkpoint found in {}".format(FLAGS.checkpoint_dir))
        print("Restoring checkpoint:", ckpt_path)
        saver.restore(sess, ckpt_path)

        writer = tf1.summary.FileWriter(FLAGS.save_summary_dir, sess.graph)

        # Run over dataset until it is exhausted
        step = 0
        try:
            while True:
                summary_str = sess.run(summary_with_updates)
                writer.add_summary(summary_str, step)
                step += 1
        except tf1.errors.OutOfRangeError:
            # NORMAL: dataset is finished
            print("Finished evaluation after {} steps (dataset exhausted)."
                  .format(step))

        # Compute final metric values (does not touch the dataset)
        final_metrics = sess.run(metrics_to_values)

        # Finalize summaries
        writer.flush()
        writer.close()

        # Print metrics to terminal
        print("Final evaluation metrics:")
        for name, value in final_metrics.items():
            print("  {}: {}".format(name, value))


if __name__ == '__main__':
    app.run(main)
