import numpy as np
from sparse_register import SparseVectorRegister


def alpha_range(x0, x1, x_min, x_max):
    """
    Helper function to calculate the 1D alpha range used in Siddon algorithm
    :param x0: start point
    :param x1: end point
    :param x_min: border start range
    :param x_max: border end range
    :return: alpha_min, alpha_max
    """
    if x0 == x1:
        raise ValueError('x1 and x2 should be different, get {} and {}'.format(x0, x1))
    alpha_x1 = (x_min - x0) / (x1 - x0)
    alpha_x2 = (x_max - x0) / (x1 - x0)
    alpha_min = max(0, min(alpha_x1, alpha_x2))
    alpha_max = min(1, max(alpha_x1, alpha_x2))
    return alpha_min, alpha_max


def alpha_interceptions(x0, x1, x_min, x_max, alpha_min, alpha_max, dim, d):
    """
    Helper function to calculate all 1D intercept points
    :param x0: start point
    :param x1: end point
    :param x_min: border start point
    :param x_max: border end point
    :param alpha_min: start alpha
    :param alpha_max: end alpha
    :param dim: number of cells
    :param d: cell size
    :return: a list of intercepted alphas
    """
    full_alpha_interceptions = (x_min + np.arange(dim + 1) * d - x0) / (x1 - x0)
    # print(full_alpha_interceptions)
    i_min = dim - (x_max - (alpha_min if x1 > x0 else alpha_max) * (x1 - x0) - x0) / d
    i_max = 1 + (x0 + (alpha_max if x1 > x0 else alpha_min) * (x1 - x0) - x_min) / d
    # print('i min {}, max {}'.format(i_min, i_max))
    alpha = full_alpha_interceptions[np.arange(i_min, i_max, dtype=np.int64)]
    # print(alpha)
    return alpha


def siddon_sparse_2d(x0, y0, x1, y1,
                     dims=[128, 128],
                     voxel_sizes=[4.0625, 4.0625]):
    """
    2D siddon algorithm
    :param x0: start x
    :param y0: start y
    :param x1: end x
    :param y1: end y
    :param dims: 2d number of cells
    :param voxel_sizes: cellsize
    :return: sparse matrix elements (i, v)
    """
    if x0 == x1 or y0 == y1:
        return siddon_special_case_sparse_2d(x0, y0, x1, y1, dims=dims, voxel_sizes=voxel_sizes)
    svr = SparseVectorRegister()
    # calculate the valid alpha range
    a_min_x, a_max_x = alpha_range(x0, x1, 0, dims[1] * voxel_sizes[1])
    a_min_y, a_max_y = alpha_range(y0, y1, 0, dims[0] * voxel_sizes[0])
    a_min = max(a_min_x, a_min_y)
    a_max = min(a_max_x, a_max_y)
    # print('a min {}, max {}'.format(a_min, a_max))
    if a_min >= a_max:
        return svr.close()
    # calculate the length of the ray
    length = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    # calculate interceptions (combine x and y interceptions)
    a_inter_x = alpha_interceptions(x0, x1, 0, dims[1] * voxel_sizes[1], a_min, a_max, dims[1], voxel_sizes[1])
    a_inter_y = alpha_interceptions(y0, y1, 0, dims[0] * voxel_sizes[0], a_min, a_max, dims[0], voxel_sizes[0])
    alpha = np.concatenate(([a_min], a_inter_x, a_inter_y, [a_max]))
    alpha = np.clip(alpha, a_min, a_max)
    alpha = np.unique(alpha)
    # calculate the sparse indices and the values
    for v in range(len(alpha) - 1):
        ix = (x0 + np.mean([alpha[v], alpha[v+1]]) * (x1 - x0) - 0)/voxel_sizes[1]
        jy = (y0 + np.mean([alpha[v], alpha[v+1]]) * (y1 - y0) - 0)/voxel_sizes[0]
        # print('ix {}, jy {}'.format(ix, jy))
        ix = int(ix)
        jy = int(jy)
        assert ix < dims[1] and jy < dims[0]
        v = length * (alpha[v+1] - alpha[v])
        svr.register(i=ix + dims[1] * jy, v=v)
    return svr.close()


def siddon_sparse_1d(x0, x1, dim, d):
    """
    1D Siddon algorithm
    """
    svr = SparseVectorRegister()
    a_min, a_max = alpha_range(x0, x1, 0, dim * d)
    if a_min >= a_max:
        return svr.close()
    length = abs(x0 - x1)
    a_inter = alpha_interceptions(x0, x1, 0, dim * d, a_min, a_max, dim, d)
    alpha = np.concatenate(([a_min], a_inter, [a_max]))
    alpha = np.clip(alpha, a_min, a_max)
    alpha = np.unique(alpha)
    for v in range(len(alpha) - 1):
        i = (x0 + np.mean([alpha[v], alpha[v+1]]) * (x1 - x0) - 0)/d
        assert i < dim
        i = int(i)
        v = length * (alpha[v+1] - alpha[v])
        svr.register(i=i, v=v)
    return svr.close()


def siddon_special_case_sparse_2d(x0, y0, x1, y1,
                                  dims=[128, 128],
                                  voxel_sizes=[4.0625, 4.0625]):
    """
    Special case siddon algorithm, either horizontal or vertical
    """
    svr = SparseVectorRegister()
    if x0 == x1:
        x = x0
        ix = x // voxel_sizes[1]
        if ix >= dims[1] or ix < 0:
            return svr.close()
        iy, v = siddon_sparse_1d(y0, y1, dim=dims[0], d=voxel_sizes[0])
        i = iy * dims[1] + ix
        return i, v
    elif y0 == y1:
        y = y0
        iy = y // voxel_sizes[0]
        if iy >= dims[0] or iy < 0:
            return svr.close()
        ix, v = siddon_sparse_1d(x0, x1, dim=dims[1], d=voxel_sizes[1])
        i = iy * dims[1] + ix
        return i, v
    else:
        raise ValueError('x, y do not satisfy special case conditions')


if __name__ == '__main__':
    siddon_sparse_2d(-1, 1, 11, 9, dims=[5,5], voxel_sizes=[2,2])
    siddon_sparse_1d(3, 11, dim=5, d=2)
    siddon_special_case_sparse_2d(-1, 0, -1, 10, dims=[5,5], voxel_sizes=[2,2])
