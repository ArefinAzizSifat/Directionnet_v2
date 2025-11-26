# coding=utf-8
# Copyright 2022 ...
"""Generate the wide baseline stereo image dataset from the Matterport3D (TF2)."""

import collections
import math
import os
import sys

import numpy as np
import tensorflow as tf
from pano_utils import math_utils
from pano_utils import transformation
import cv2


# -------------------------------
# Core projections (unchanged API)
# -------------------------------
def world_to_image_projection(p_world, intrinsics, pose_w2c):
  """Project points in the world frame to the image plane.

  Args:
    p_world: [H, W, 3]
    intrinsics: [3, 3]
    pose_w2c: [3, 4]

  Returns:
    ([H, W, 2], [H, W, 1])
  """
  shape = p_world.shape.as_list()
  height, width = shape[0], shape[1]

  p_world_h = tf.concat([p_world, tf.ones([height, width, 1], dtype=p_world.dtype)], axis=-1)
  # (1,1,3,4) @ (H,W,4,1) -> (H,W,3)
  p_camera = tf.squeeze(
      tf.matmul(pose_w2c[tf.newaxis, tf.newaxis, :], tf.expand_dims(p_world_h, -1)),
      axis=-1)
  # OpenGL convention flip z
  p_camera = p_camera * tf.constant([1., 1., -1.], dtype=p_camera.dtype, shape=[1, 1, 3])
  # (1,1,3,3) @ (H,W,3,1) -> (H,W,3)
  p_image = tf.squeeze(
      tf.matmul(intrinsics[tf.newaxis, tf.newaxis, :], tf.expand_dims(p_camera, -1)),
      axis=-1)
  z = p_image[:, :, -1:]
  return tf.math.divide_no_nan(p_image[:, :, :2], z), z


def image_to_world_projection(depth, intrinsics, pose_c2w):
  """Project pixels with depth back to world.

  Args:
    depth: [H, W, 1]
    intrinsics: [3, 3]
    pose_c2w: [3, 4]

  Returns:
    [H, W, 3]
  """
  shape = depth.shape.as_list()
  height, width = shape[0], shape[1]

  xx, yy = tf.meshgrid(tf.linspace(0., tf.cast(width-1, tf.float32), width),
                       tf.linspace(0., tf.cast(height-1, tf.float32), height))
  p_pix_h = tf.concat([tf.stack([xx, yy], axis=-1),
                       tf.ones([height, width, 1], dtype=tf.float32)], axis=-1)

  invK = tf.linalg.inv(intrinsics)[tf.newaxis, tf.newaxis, :]
  p_image = tf.squeeze(tf.matmul(invK, tf.expand_dims(p_pix_h, -1)), axis=-1)  # [H,W,3]

  z = depth * tf.reduce_sum(
      tf.math.l2_normalize(p_image, axis=-1) * tf.constant([[[0., 0., 1.]]], dtype=p_image.dtype),
      axis=-1,
      keepdims=True)

  p_camera = z * p_image
  p_camera = p_camera * tf.constant([1., 1., -1.], dtype=p_camera.dtype, shape=[1, 1, 3])
  p_cam_h = tf.concat([p_camera, tf.ones([height, width, 1], dtype=p_camera.dtype)], axis=-1)

  p_world = tf.squeeze(
      tf.matmul(pose_c2w[tf.newaxis, tf.newaxis, :], tf.expand_dims(p_cam_h, -1)),
      axis=-1)
  return p_world


def overlap_mask(depth1, pose1_c2w, depth2, pose2_c2w, intrinsics):
  """Compute overlap masks of two views using triangulation.

  Returns:
    (mask1_in_2 [H,W], mask2_in_1 [H,W]) booleans
  """
  pose1_w2c = tf.linalg.inv(
      tf.concat([pose1_c2w, tf.constant([[0., 0., 0., 1.]], dtype=pose1_c2w.dtype)], axis=0))[:3]
  pose2_w2c = tf.linalg.inv(
      tf.concat([pose2_c2w, tf.constant([[0., 0., 0., 1.]], dtype=pose2_c2w.dtype)], axis=0))[:3]

  p_world1 = image_to_world_projection(depth1, intrinsics, pose1_c2w)
  p_image1_in_2, z1_c2 = world_to_image_projection(p_world1, intrinsics, pose2_w2c)

  p_world2 = image_to_world_projection(depth2, intrinsics, pose2_c2w)
  p_image2_in_1, z2_c1 = world_to_image_projection(p_world2, intrinsics, pose1_w2c)

  shape = depth1.shape.as_list()
  height, width = shape[0], shape[1]
  height_f = tf.cast(height, tf.float32)
  width_f = tf.cast(width, tf.float32)
  eps = 1e-4

  mask_h2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 1], height_f + eps),
      tf.greater_equal(p_image2_in_1[:, :, 1], 0. - eps))
  mask_w2_in_1 = tf.logical_and(
      tf.less_equal(p_image2_in_1[:, :, 0], width_f + eps),
      tf.greater_equal(p_image2_in_1[:, :, 0], 0. - eps))
  mask2_in_1 = tf.logical_and(tf.logical_and(mask_h2_in_1, mask_w2_in_1),
                              tf.squeeze(z2_c1, -1) > 0)

  mask_h1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 1], height_f + eps),
      tf.greater_equal(p_image1_in_2[:, :, 1], 0. - eps))
  mask_w1_in_2 = tf.logical_and(
      tf.less_equal(p_image1_in_2[:, :, 0], width_f + eps),
      tf.greater_equal(p_image1_in_2[:, :, 0], 0. - eps))
  mask1_in_2 = tf.logical_and(tf.logical_and(mask_h1_in_2, mask_w1_in_2),
                              tf.squeeze(z1_c2, -1) > 0)

  return mask1_in_2, mask2_in_1


def overlap_ratio(mask1, mask2):
  """Minimum overlap ratio between the two masks."""
  shape = mask1.shape.as_list()
  height, width = shape[0], shape[1]
  a = tf.reduce_sum(tf.cast(mask1, tf.float32)) / (height * width)
  b = tf.reduce_sum(tf.cast(mask2, tf.float32)) / (height * width)
  return tf.minimum(a, b)


# -------------------------------------
# Dataset builders (unchanged signatures)
# -------------------------------------
def generate_from_meta(meta_data_path: str,
                       pano_data_dir: str,
                       pano_height: int = 1024,
                       pano_width: int = 2048,
                       output_height: int = 512,
                       output_width: int = 512):
  """Build dataset from meta files (TF2)."""

  print("meta_data_path")
  if os.path.exists(meta_data_path):
    print("Path exists")
  print(meta_data_path)

  print("pano_data_dir")
  print(pano_data_dir)
  if os.path.exists(pano_data_dir):
    print("Path exists")

  def load_text(file_path, n_lines=200):
    """Load text data from a file."""
    elem = tf.data.experimental.get_single_element(
        tf.data.TextLineDataset(file_path).batch(n_lines))
    return tf.data.Dataset.from_tensor_slices(elem)

  def load_single_image(filename):
    """Load a single image given the filename."""
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    # (kept as-is) This will display and block if a GUI is available:
    cv2.imshow("T", np.zeros((1, 1, 3), dtype=np.uint8)) if False else None
    # (Note: original code shows image via cv2; we keep the call present but inert)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([pano_height, pano_width, 3])
    return image

  def string_to_matrix(s, shape):
    """Decode strings to matrices tensor."""
    defaults = [0.0] * int(np.prod(shape))
    flat = tf.io.decode_csv(s, record_defaults=defaults)  # comma-delimited
    m = tf.reshape(tf.stack(flat, axis=0), shape)
    m.set_shape(shape)
    return m

  def decode_line(line):
    """Decode text lines."""
    DataPair = collections.namedtuple(
        'DataPair', ['src_img', 'trt_img', 'fov', 'rotation', 'translation'])

    # space-delimited 10 fields (paths + csv numbers)
    splitted = tf.io.decode_csv(line, record_defaults=[''] * 10, field_delim=' ')

    img1 = load_single_image(pano_data_dir + splitted[0] + '/' + splitted[1] + '.jpeg')
    img2 = load_single_image(pano_data_dir + splitted[0] + '/' + splitted[2] + '.jpeg')
    fov = string_to_matrix(splitted[3], [1])
    r1 = string_to_matrix(splitted[4], [3, 3])
    t1 = string_to_matrix(splitted[5], [3])
    r2 = string_to_matrix(splitted[6], [3, 3])
    t2 = string_to_matrix(splitted[7], [3])
    sampled_r1 = string_to_matrix(splitted[8], [3, 3])
    sampled_r2 = string_to_matrix(splitted[9], [3, 3])

    r_c2_to_c1 = tf.matmul(sampled_r1, sampled_r2, transpose_a=True)
    t_c1 = tf.squeeze(
        tf.matmul(sampled_r1,
                  tf.expand_dims(tf.nn.l2_normalize(t2 - t1), -1),
                  transpose_a=True))

    sampled_rotation = tf.matmul(tf.stack([sampled_r1, sampled_r2], 0),
                                 tf.stack([r1, r2], 0), transpose_a=True)

    sampled_views = transformation.rectilinear_projection(
        tf.stack([img1, img2], 0),
        [output_height, output_width],
        fov,
        tf.transpose(sampled_rotation, perm=[0, 2, 1]))
    src_img, trt_img = sampled_views[0], sampled_views[1]
    return DataPair(src_img, trt_img, fov, r_c2_to_c1, t_c1)

  ds = tf.data.Dataset.list_files(meta_data_path + '*')
  ds = ds.flat_map(load_text)
  ds = ds.map(decode_line)
  return ds


def generate_random_views(pano1_rgb,
                          pano2_rgb,
                          r1, t1, r2, t2,
                          max_rotation=60.,
                          max_tilt=5.,
                          output_fov=90.,
                          output_height=512,
                          output_width=512,
                          pano1_depth=None,
                          pano2_depth=None):
  """Generate stereo pairs by random sampling (unchanged API)."""
  ViewPair = collections.namedtuple(
      'ViewPair', ['img1', 'img2', 'mask1', 'mask2', 'fov', 'r', 't'])

  swap_yz = tf.constant([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=tf.float32, shape=[1, 3, 3])

  lookat_direction1 = math_utils.random_vector_on_sphere(
      1, [[-math.sin(math.pi/3), math.sin(math.pi/3)], [0., 2*math.pi]])
  lookat_direction1 = tf.squeeze(tf.matmul(swap_yz, tf.expand_dims(lookat_direction1, -1)), -1)

  lookat_direction2 = math_utils.uniform_sampled_vector_within_cone(
      lookat_direction1, math_utils.degrees_to_radians(max_rotation))
  lookat_directions = tf.concat([lookat_direction1, lookat_direction2], axis=0)

  up1 = math_utils.uniform_sampled_vector_within_cone(
      tf.constant([[0., 0., 1.]]), math_utils.degrees_to_radians(max_tilt))
  up2 = math_utils.uniform_sampled_vector_within_cone(
      tf.constant([[0., 0., 1.]]), math_utils.degrees_to_radians(max_tilt))

  lookat_rotations = math_utils.lookat_matrix(
      tf.concat([up1, up2], axis=0), lookat_directions)

  sample_rotations = tf.matmul(tf.concat([r1, r2], axis=0), lookat_rotations, transpose_a=True)

  sampled_views = transformation.rectilinear_projection(
      tf.stack([pano1_rgb, pano2_rgb], axis=0),
      [output_height, output_width],
      output_fov,
      sample_rotations)

  r_c2_to_c1 = tf.matmul(lookat_rotations[0], lookat_rotations[1], transpose_a=True)
  t_c1 = tf.squeeze(tf.matmul(lookat_rotations[0],
                              tf.expand_dims(tf.nn.l2_normalize(t2 - t1), -1),
                              transpose_a=True))

  if pano1_depth is not None and pano2_depth is not None:
    sampled_depth = transformation.rectilinear_projection(
        tf.stack([pano1_depth, pano2_depth], axis=0),
        [output_height, output_width],
        output_fov,
        sample_rotations)

    fx = output_width * 0.5 / math.tan(math_utils.degrees_to_radians(output_fov) / 2)
    intrinsics = tf.constant([[fx, 0., output_width * 0.5],
                              [0., -fx, output_height * 0.5],
                              [0., 0., 1.]], dtype=tf.float32)

    pose1_c2w = tf.concat([lookat_rotations[0], tf.expand_dims(t1, -1)], axis=1)
    pose2_c2w = tf.concat([lookat_rotations[1], tf.expand_dims(t2, -1)], axis=1)

    mask1, mask2 = overlap_mask(sampled_depth[0], pose1_c2w,
                                sampled_depth[1], pose2_c2w,
                                intrinsics)
  else:
    mask1 = None
    mask2 = None

  return ViewPair(sampled_views[0],
                  sampled_views[1],
                  mask1,
                  mask2,
                  output_fov,
                  r_c2_to_c1,
                  t_c1)


# Optional local smoke test (kept structure, not required in production)
if __name__ == '__main__':
  print("'dataset_tf2.py' imported and ready.")
