import numpy as np
import torch
import pytest

from nn.common.rotation_transformer import RotationTransformer


def test_axis_angle_rotation6d_roundtrip_numpy():
    tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    rotvec = np.random.uniform(-np.pi, np.pi, size=(100, 3)).astype(np.float32)
    rot6d = tf.forward(rotvec)
    recovered = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(recovered).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-5


def test_axis_angle_rotation6d_roundtrip_torch():
    # Avoid angles near ±π where axis-angle representation is ambiguous
    tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    rotvec = (torch.rand(50, 3) * 2 - 1) * (np.pi * 0.8)
    rot6d = tf.forward(rotvec)
    recovered = tf.inverse(rot6d)
    assert torch.allclose(rotvec, recovered, atol=1e-5)


def test_rotation6d_to_matrix_det_one():
    tf_gen = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    tf_mat = RotationTransformer(from_rep="rotation_6d", to_rep="matrix")

    rotvec = np.random.uniform(-np.pi, np.pi, size=(50, 3)).astype(np.float32)
    rot6d = tf_gen.forward(rotvec)
    # Perturb to test normalization
    rot6d_noisy = rot6d + np.random.normal(scale=0.1, size=rot6d.shape).astype(np.float32)
    mat = tf_mat.forward(rot6d_noisy)
    det = np.linalg.det(mat)
    assert np.allclose(det, 1.0, atol=1e-5)


def test_numpy_and_torch_inputs_consistent():
    tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    rotvec_np = np.random.uniform(-1.0, 1.0, size=(10, 3)).astype(np.float32)
    rotvec_torch = torch.from_numpy(rotvec_np)

    out_np = tf.forward(rotvec_np)
    out_torch = tf.forward(rotvec_torch)

    assert np.allclose(out_np, out_torch.numpy(), atol=1e-6)
