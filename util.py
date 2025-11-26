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
"""Utilities for math and geometry operations (TF2-compatible)."""
import math
import pickle
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
# Keep your original module structure/names
from pano_utils import geometry
from pano_utils import math_utils
from pano_utils import transformation


def read_pickle(file):
    print(file)
    """Read from pickle files."""
    with open(file, 'rb') as f:
        loaded = pickle.load(f, encoding='bytes')
        print("Loaded =============================")
    return list(loaded.keys()), list(loaded.values())


def safe_sqrt(x):
    return tf.sqrt(tf.maximum(x, 1e-20))


def degrees_to_radians(degree):
    """Convert degrees to radians."""
    return math.pi * degree / 180.0


def radians_to_degrees(radians):
    """Convert radians to degrees."""
    return 180.0 * radians / math.pi


def angular_distance(v1, v2):
    dot = tf.reduce_sum(v1 * v2, -1)
    return tf.acos(tf.clip_by_value(dot, -1., 1.))


def equirectangular_area_weights(height):
    """Generate area weights for pixels in equirectangular images.
    Args:
      height: the height dimension of the equirectangular images.
    Returns:
      Area weights with shape [1, HEIGHT, 1, 1].
    """
    with tf.name_scope('equirectangular_area_weights'):
        pixel_h = math.pi / tf.cast(height, tf.float32)
        # Use half-integer pixel centre convention.
        colatitude = tf.linspace(pixel_h / 2, math.pi - pixel_h / 2, height)
        colatitude = colatitude[tf.newaxis, :, tf.newaxis, tf.newaxis]
        return tf.sin(colatitude)


def spherical_normalization(x, rectify=True):
    """Apply area weights and normalization to spherical distributions.
    The sum of all pixel values over the spherical input will be one.
    Args:
      x: [BATCH, HEIGHT, WIDTH, CHANNELS] spherical raw distributions.
      rectify: apply softplus to the input x if true.
    Returns:
      [BATCH, HEIGHT, WIDTH, CHANNELS] normalized distributions.
    """
    with tf.name_scope('spherical_normalization'):
        shape = x.shape.as_list()
        height = shape[1]
        if rectify:
            x = tf.nn.softplus(x)
        weighted = x * equirectangular_area_weights(height)
        # Return shape [BATCH, HEIGHT, WIDTH, CHANNELS].
        denom = tf.reduce_sum(weighted, axis=[1, 2], keepdims=True)
        return tf.math.divide_no_nan(x, denom)


def spherical_expectation(spherical_probabilities):
    """Compute the expectation (a vector) from normalized spherical distributions.
    Args:
      spherical_probabilities: [BATCH, HEIGHT, WIDTH, N] (normalized).
    Returns:
      expectation [BATCH, N, 3]
    """
    shape = spherical_probabilities.shape.as_list()
    height, width, channels = shape[1], shape[2], shape[3]
    spherical = tf.expand_dims(
        geometry.generate_equirectangular_grid([height, width]), 0)
    unit_directions = geometry.spherical_to_cartesian(spherical)
    axis_convert = tf.constant(
        [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=unit_directions.dtype)
    unit_directions = tf.squeeze(
        tf.matmul(axis_convert, tf.expand_dims(unit_directions, -1), transpose_a=True), -1)
    unit_directions = tf.tile(
        tf.expand_dims(unit_directions, -2), [1, 1, 1, channels, 1])
    weighted = spherical_probabilities * equirectangular_area_weights(height)
    expectation = tf.reduce_sum(
        unit_directions * tf.expand_dims(weighted, -1), [1, 2])
    return expectation


def von_mises_fisher(mean, concentration, shape):
    """Generate von Mises-Fisher distribution on spheres.
    Args:
      mean: [BATCH, N, 3] unit mean directions.
      concentration: (float) kappa.
      shape: [HEIGHT, WIDTH].
    Returns:
      [BATCH, HEIGHT, WIDTH, N] raw (unnormalized surface integral) probabilities.
    """
    with tf.name_scope('von_mises_fisher'):
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("Input argument 'shape' is not valid.")
        if mean.shape[-1] != 3:
            raise ValueError("Input argument 'mean' has wrong dimensions.")
        mean_shape = tf.shape(mean)
        batch = mean_shape[0]
        channels = mean_shape[1]
        height, width = shape[0], shape[1]
        spherical_grid = geometry.generate_equirectangular_grid(shape)
        cartesian = geometry.spherical_to_cartesian(spherical_grid)
        axis_convert = tf.constant(
            [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=cartesian.dtype)
        cartesian = tf.squeeze(
            tf.matmul(axis_convert, tf.expand_dims(cartesian, -1), transpose_a=True), -1)
        cartesian = tf.tile(
            cartesian[tf.newaxis, tf.newaxis, :],
            tf.stack([batch, channels, 1, 1, 1]))
        mean = tf.tile(mean[:, :, tf.newaxis, tf.newaxis],
                       [1, 1, height, width, 1])
        tfd = tfp.distributions
        vmf = tfd.VonMisesFisher(
            mean_direction=mean, concentration=[concentration])
        spherical_gaussian = vmf.prob(cartesian)  # [B,N,H,W]
        return tf.transpose(spherical_gaussian, [0, 2, 3, 1])


def rotation_geodesic(r1, r2):
    """Return the geodesic distance (angle in radians) between two rotations.
    Args:
      r1: [BATCH, 3, 3] rotation matrices.
      r2: [BATCH, 3, 3] rotation matrices.
    Returns:
      [BATCH] radian angular difference.
    """
    diff = (tf.linalg.trace(tf.matmul(r1, r2, transpose_b=True)) - 1.0) / 2.0
    angular_diff = tf.acos(tf.clip_by_value(diff, -1., 1.))
    return angular_diff


def gram_schmidt(m):
    """Convert 6D representation to SO(3) using a partial Gram-Schmidt process.
    Args:
      m: [BATCH, 2, 3] 2x3 matrices.
    Returns:
      [BATCH, 3, 3] SO(3) rotation matrices.
    """
    x = m[:, 0]
    y = m[:, 1]
    xn = tf.math.l2_normalize(x, axis=-1)
    z = tf.linalg.cross(xn, y)
    zn = tf.math.l2_normalize(z, axis=-1)
    y = tf.linalg.cross(zn, xn)
    r = tf.stack([xn, y, zn], 1)
    return r


def svd_orthogonalize(m):
    """Convert 9D representation to SO(3) using SVD orthogonalization.
    Args:
      m: [BATCH, 3, 3] 3x3 matrices.
    Returns:
      [BATCH, 3, 3] SO(3) rotation matrices.
    """
    m_transpose = tf.transpose(
        tf.math.l2_normalize(m, axis=-1), perm=[0, 2, 1])
    s, u, v = tf.linalg.svd(m_transpose, full_matrices=False)
    det = tf.linalg.det(tf.matmul(v, u, transpose_b=True))
    v_last = v[:, :, -1:] * tf.reshape(det, [-1, 1, 1])
    v_fixed = tf.concat([v[:, :, :-1], v_last], axis=2)
    r = tf.matmul(v_fixed, u, transpose_b=True)
    return r


def perturb_rotation(r, perturb_limits):
    """Randomly perturb a 3d rotation with a normal distribution.
    Args:
      r: [BATCH, 3, 3] rotation matrices.
      perturb_limits: a 3d list (degrees) for axes x, y, z.
    Returns:
      [BATCH, 3, 3] perturbed rotation matrices.
    """
    x, y, z = tf.split(r, [1, 1, 1], 1)
    x = math_utils.normal_sampled_vector_within_cone(
        tf.squeeze(x, 1), degrees_to_radians(perturb_limits[0]), 0.5)
    y = math_utils.normal_sampled_vector_within_cone(
        tf.squeeze(y, 1), degrees_to_radians(perturb_limits[1]), 0.5)
    z = math_utils.normal_sampled_vector_within_cone(
        tf.squeeze(z, 1), degrees_to_radians(perturb_limits[2]), 0.5)
    return svd_orthogonalize(tf.stack([x, y, z], 1))


def half_rotation(rotation):
    """Return half of the input rotation.
    Args:
      rotation: [BATCH, 3, 3] rotation matrices.
    Returns:
      [BATCH, 3, 3] rotation matrices.
    """
    axes, angles = axis_angle.from_rotation_matrix(rotation)
    return rotation_matrix_3d.from_axis_angle(axes, angles / 2.0)


def distributions_to_directions(x):
    """Convert spherical distributions from the DirectionNet to directions."""
    distribution_pred = spherical_normalization(x)
    expectation = spherical_expectation(distribution_pred)
    expectation_normalized = tf.nn.l2_normalize(expectation, axis=-1)
    return expectation_normalized, expectation, distribution_pred


def derotation(src_img,
               trt_img,
               rotation,
               input_fov,
               output_fov,
               output_shape,
               derotate_both):
    """Transform a pair of images to cancel out the rotation.
    Args:
      src_img: [BATCH, HEIGHT, WIDTH, CHANNEL] input source images.
      trt_img: [BATCH, HEIGHT, WIDTH, CHANNEL] input target images.
      rotation: [BATCH, 3, 3] relative rotations between src_img and trt_img.
      input_fov: [BATCH] float32 degrees.
      output_fov: (float) degrees.
      output_shape: [height, width].
      derotate_both: bool.
    Returns:
      (transformed_src, transformed_trt): both [BATCH, h, w, C].
    """
    batch = src_img.shape.as_list()[0]
    if derotate_both:
        half_derotation = half_rotation(rotation)
        transformed_src = transformation.rotate_image_in_3d(
            src_img,
            tf.transpose(half_derotation, perm=[0, 2, 1]),
            input_fov,
            output_fov,
            output_shape)
        transformed_trt = transformation.rotate_image_in_3d(
            trt_img,
            half_derotation,
            input_fov,
            output_fov,
            output_shape)
    else:
        transformed_src = transformation.rotate_image_in_3d(
            src_img,
            tf.eye(3, batch_shape=[batch], dtype=src_img.dtype),
            input_fov,
            output_fov,
            output_shape)
        transformed_trt = transformation.rotate_image_in_3d(
            trt_img,
            rotation,
            input_fov,
            output_fov,
            output_shape)

    return transformed_src, transformed_trt
