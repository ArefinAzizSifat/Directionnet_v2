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
"""Transformations for equirectangular and perspective images.
The coordinate system is the same as OpenGL's, where -Z is the camera looking
direction, +Y points up and +X points right.
Rotations are applied as pre-multiplication in all cases.
"""
import math
import tensorflow as tf
import tensorflow_addons as tfa

from pano_utils import geometry
from pano_utils import math_utils


def equirectangular_sampler(images, spherical_coordinates):
    """Sample panorama images using a grid of spherical coordinates.
    Args:
      images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
      spherical_coordinates: a float32 tensor with shape
        [BATCH, sampling_height, sampling_width, 2] representing spherical
        coordinates (colatitude, azimuth) of the sampling grids.
    Returns:
      a 4-D tensor of shape `[BATCH, sampling_height, sampling_width, CHANNELS]`
      representing resampled images.
    Raises:
      ValueError: 'images' or 'spherical_coordinates' has the wrong dimensions.
    """
    if len(images.shape) != 4:
        raise ValueError("'images' has the wrong dimensions.")
    if spherical_coordinates.shape[-1] != 2:
        raise ValueError("'spherical_coordinates' has the wrong dimensions.")
    shape = images.shape.as_list()
    height, width = shape[1], shape[2]
    padded_images = geometry.equirectangular_padding(images, [[1, 1], [1, 1]])
    colatitude, azimuth = tf.split(spherical_coordinates, [1, 1], axis=-1)
    # Convert spherical to pano coords (half-pixel convention) +1 for padding
    x_pano = (tf.math.floormod(azimuth / math.pi, 2.0)
              * (width / 2.0) - 0.5) + 1.0
    y_pano = ((colatitude / math.pi) * height - 0.5) + 1.0
    pano_coordinates = tf.concat([x_pano, y_pano], axis=-1)  # [B,h,w,2] (x,y)
    remapped = tfa.image.resampler(padded_images, pano_coordinates)
    return remapped


def rectilinear_projection(images, resolution, fov, rotations):
    """Convert equirectangular panoramic images to perspective images.
    First, the panorama images are rotated by the input parameter "rotations".
    Then, the region with the field of view "fov" centered at camera's look-at -Z
    axis is projected into perspective images. The -Z axis corresponds to the
    spherical coordinates (pi/2, pi/2) which is (HEIGHT/2, WIDTH/4) on the pano.
    Args:
      images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
      resolution: a 2-D tuple or list containing the resolution of desired output.
      fov: (float) camera's horizontal field of view in degrees.
      rotations: [BATCH, 3, 3] rotation matrices.
    Returns:
      4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`
    """
    if len(images.shape) != 4:
        raise ValueError("'images' has the wrong dimensions.")
    if images.dtype not in (tf.float32, tf.float64):
        raise ValueError("'images' must be a float tensor.")
    if rotations.shape[-2:] != [3, 3]:
        raise ValueError("'rotations' has the wrong dimensions.")
    shape = images.shape.as_list()
    batch = shape[0]
    cartesian_coordinates = geometry.generate_cartesian_grid(
        resolution, fov)      # [H,W,3]
    cartesian_coordinates = tf.tile(tf.expand_dims(cartesian_coordinates, 0),
                                    # [B,H,W,3]
                                    [batch, 1, 1, 1])
    flip_x = tf.constant([[-1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]], dtype=images.dtype)
    rotations = tf.matmul(flip_x, tf.matmul(
        rotations, flip_x, transpose_a=True))  # [B,3,3]
    # [B,3,3] @ [B,H,W,3,1] -> [B,H,W,3,1]
    rotated = tf.matmul(
        rotations[:, tf.newaxis, tf.newaxis, :, :],
        cartesian_coordinates[..., tf.newaxis],
        # [B,H,W,3,1]
        transpose_a=True)
    axis_convert = tf.constant([[0., 0., 1.],
                                [1., 0., 0.],
                                [0., 1., 0.]], dtype=images.dtype)
    # [B,H,W,3,1]
    rotated = tf.matmul(axis_convert, rotated)
    rotated_coordinates = tf.squeeze(
        rotated, axis=-1)                             # [B,H,W,3]
    spherical_coordinates = geometry.cartesian_to_spherical(
        rotated_coordinates)   # [B,H,W,2]
    spherical_coordinates = tf.reverse(spherical_coordinates, axis=[
                                       2])            # [B,H,W,2]
    return equirectangular_sampler(images, spherical_coordinates)


def rotate_pano(images, rotations):
    """Rotate Panoramic images.
    Convert the spherical coordinates (colatitude, azimuth) to Cartesian (x, y, z)
    then apply SO(3) rotation matrices. Finally, convert them back to spherical
    coordinates and remap the equirectangular images.
    Note1: The rotations are applied to the sampling sphere instead of the camera.
    The camera actually rotates R^T. I_out(x) = I_in(R * x), x are points in the
    camera frame.
    Note2: It uses a simple linear interpolation for now instead of slerp, so the
    pixel values are not accurate but visually plausible.
    Args:
      images: a 4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
      rotations: [BATCH, 3, 3] rotation matrices.
    Returns:
      4-D tensor of shape `[BATCH, HEIGHT, WIDTH, CHANNELS]`.
    """
    if len(images.shape) != 4:
        raise ValueError("'images' has the wrong dimensions.")
    if rotations.shape[-2:] != [3, 3]:
        raise ValueError("'rotations' must have 3x3 dimensions.")
    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    spherical = tf.expand_dims(
        geometry.generate_equirectangular_grid([height, width]), 0)
    # [B,H,W,2]
    spherical = tf.tile(spherical, [batch, 1, 1, 1])
    cartesian = geometry.spherical_to_cartesian(
        spherical)                         # [B,H,W,3]
    axis_convert = tf.constant([[0.,  1.,  0.],
                                [0.,  0., -1.],
                                [-1.,  0.,  0.]], dtype=images.dtype)
    # [B,H,W,3,1]
    cartesian = tf.matmul(axis_convert, cartesian[..., tf.newaxis])
    rotated_cartesian = tf.matmul(
        rotations[:, tf.newaxis, tf.newaxis, :, :], cartesian)
    rotated_cartesian = tf.squeeze(
        # [B,H,W,3]
        tf.matmul(axis_convert, rotated_cartesian, transpose_a=True), axis=-1)
    rotated_spherical = geometry.cartesian_to_spherical(
        rotated_cartesian)         # [B,H,W,2]
    return equirectangular_sampler(images, rotated_spherical)


def rotate_image_in_3d(images, input_rotations, input_fov, output_fov, output_shape):
    """Return reprojected perspective view images given a rotated camera.
    This applies H = K_out * R^T * K_in^{-1} (implicitly via normalized plane).
    Args:
      images: [BATCH, HEIGHT, WIDTH, CHANNEL] perspective view images.
      input_rotations: [BATCH, 3, 3] rotations matrices from current camera frame
        to target camera frame.
      input_fov: [BATCH] float32 of input field of view in degrees.
      output_fov: (float) output field of view in degrees.
      output_shape: [height, width] of output.
    """
    if len(images.shape) != 4:
        raise ValueError("'images' has the wrong dimensions.")
    if input_rotations.shape[-2:] != [3, 3]:
        raise ValueError("'input_rotations' must have 3x3 dimensions.")
    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    cartesian = geometry.generate_cartesian_grid(
        output_shape, output_fov)         # [h,w,3]
    cartesian = tf.tile(cartesian[tf.newaxis, :, :, :, tf.newaxis], [
                        batch, 1, 1, 1, 1])  # [B,h,w,3,1]
    input_rotations = tf.tile(input_rotations[:, tf.newaxis, tf.newaxis, :, :],
                              # [B,h,w,3,3]
                              [1] + output_shape + [1, 1])
    cartesian = tf.squeeze(
        # [B,h,w,3]
        tf.matmul(input_rotations, cartesian, transpose_a=True), axis=-1)
    image_coordinates = -cartesian[:, :, :, :2] / \
        cartesian[:, :, :, -1:]          # [B,h,w,2]
    x, y = tf.split(image_coordinates, [1, 1], axis=-1)
    # [B]
    w = 2.0 * tf.tan(math_utils.degrees_to_radians(input_fov / 2.0))
    # [B]
    h = 2.0 * tf.tan(math_utils.degrees_to_radians(input_fov / 2.0))
    # [B,1,1,1]
    w = w[:, tf.newaxis, tf.newaxis, tf.newaxis]
    h = h[:, tf.newaxis, tf.newaxis, tf.newaxis]
    nx = x * (width / w) + (width / 2.0) - 0.5
    ny = -y * (height / h) + (height / 2.0) - 0.5
    return tfa.image.resampler(images, tf.concat([nx, ny], axis=-1))


def rotate_image_on_pano(images, rotations, fov, output_shape):
    """Transform perspective images to equirectangular images after rotations.
    Return equirectangular panoramic images in which the input perspective images
    are embedded after rotation R from the input frame to target frame. The image
    with field of view "fov" centered at camera's look-at -Z axis is projected
    onto the pano. The -Z axis corresponds to the spherical coordinates
    (pi/2, pi/2) which is (HEIGHT/2, WIDTH/4) on the pano.
    """
    if len(images.shape) != 4:
        raise ValueError("'images' has the wrong dimensions.")
    if rotations.shape[-2:] != [3, 3]:
        raise ValueError("'rotations' must have 3x3 dimensions.")
    shape = images.shape.as_list()
    batch, height, width = shape[0], shape[1], shape[2]
    # Mesh on sphere in output pano
    spherical = geometry.generate_equirectangular_grid(
        output_shape)               # [h,w,2]
    cartesian = geometry.spherical_to_cartesian(
        spherical)                         # [h,w,3]
    cartesian = tf.tile(cartesian[tf.newaxis, :, :, :, tf.newaxis], [
                        batch, 1, 1, 1, 1])  # [B,h,w,3,1]
    axis_convert = tf.constant([[0., -1.,  0.],
                                [0.,  0.,  1.],
                                [1.,  0.,  0.]], dtype=images.dtype)
    # [B,h,w,3,1]
    cartesian = tf.matmul(axis_convert, cartesian)
    cartesian = tf.squeeze(tf.matmul(
        # [B,h,w,3]
        rotations[:, tf.newaxis, tf.newaxis, :, :], cartesian), axis=-1)
    hemisphere_mask = tf.cast(
        cartesian[:, :, :, -1:] < 0.0, images.dtype)         # [B,h,w,1]
    image_coordinates = cartesian[:, :, :, :2] / \
        cartesian[:, :, :, -1:]           # [B,h,w,2]
    x, y = tf.split(image_coordinates, [1, 1], axis=-1)
    denom = 2.0 * tf.tan(math_utils.degrees_to_radians(fov / 2.0))
    nx = -x * (width / denom) + (width / 2.0) - 0.5
    ny = y * (height / denom) + (height / 2.0) - 0.5
    transformed = hemisphere_mask * \
        tfa.image.resampler(images, tf.concat([nx, ny], axis=-1))
    return transformed
