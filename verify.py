# https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/19
import torch
import torch.nn.functional as F
import cv2
import numpy as np


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`

    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image

    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def cvt_ThetaToM(theta, w, h, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    theta_aug = np.concatenate([theta, np.zeros((1, 3))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv[:2, :]
    return M[:2, :]


if __name__ == '__main__':
    input_numpy = cv2.imread('./data/inoki_640x640.JPG')

    w = input_numpy.shape[1]
    h = input_numpy.shape[0]
    M = np.random.rand(2, 3)

    # Convert M -> theta -> M2
    # Verify M == M2
    theta = cvt_MToTheta(M, w, h)
    M2 = cvt_ThetaToM(theta, w, h)
    assert np.allclose(M, M2)

    output_numpy = cv2.warpAffine(input_numpy, M, (input_numpy.shape[1], input_numpy.shape[0]))
    cv2.imwrite('./data/output_numpy.png', output_numpy)

    input_torch = torch.from_numpy(input_numpy.astype(np.float32)).clone().permute(2, 0, 1).unsqueeze(0)
    input_torch = input_torch / 255.0
    theta = torch.from_numpy(theta.astype(np.float32)).clone().unsqueeze(0)
    grid = F.affine_grid(theta, input_torch.size())
    output_torch = F.grid_sample(input_torch, grid).to('cpu').squeeze().detach().numpy().copy()
    output_torch = output_torch.transpose(1, 2, 0) * 255.0
    output_torch = output_torch.astype(np.uint8)
    cv2.imwrite('./data/output_torch.png', output_torch)

    diff = output_numpy - output_torch
    print('diff max: {}'.format(diff.max()))
    cv2.imwrite('./data/diff.png', diff)

    assert np.allclose(output_torch, output_numpy)
