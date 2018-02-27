import numpy as np
import scipy.sparse as sparse
from siddon import siddon_sparse_2d
# import argparse


def construct_system_matrix_single_angle(a, d,
                                         prj_dims=[128, 128],
                                         img_dims=[128, 128, 128],
                                         prj_cellsize=[3.125, 4.0625],
                                         img_cellsize=[3.125, 4.0625, 4.0625]):
    """
    Construct the system matrix for a specified angle, also given its init distance

    The 3D system matrix can be viewed as a replication of multiple 2D system matrix.
    For each layer along the z-dim, we replicate the 2D matrix calculated from the first layer, and then concat
    them along the row dim.
    :param a: the angle [0, pi)
    :param d: the distance between detector and the image center
    :param prj_dims: projection dimensions, (height, width)
    :param img_dims: image dimensions, (layer, height, width)
    :param prj_cellsize: detector unit size
    :param img_cellsize: image voxel size
    :return: elements of the sparse matrix, (ir, jc, value, size of matrix)
    """
    if prj_dims[0] != img_dims[0]:
        raise NotImplementedError
    print('Construct angle {}, distance {} mm'.format(a, d))
    mat_dims = (np.prod(prj_dims), np.prod(img_dims))
    img_height = img_dims[1] * img_cellsize[1]
    img_width = img_dims[2] * img_cellsize[2]
    prj_width = prj_dims[1] * prj_cellsize[1]
    ir_full = list()
    jc_full = list()
    v_full = list()
    # for each detector unit along the first row
    for i_width in range(prj_dims[1]):
        r = (i_width + 0.5) * img_cellsize[2]
        # the line function should be: cos(a)(y - y0) = - sin(a)(x - x0)
        x0 = img_width / 2 - d * np.cos(a) - (prj_width / 2 - r) * np.sin(a)
        y0 = img_height / 2 + d * np.sin(a) - (prj_width / 2 - r) * np.cos(a)
        # switch angle `a` to determine the end point (x1, y1)
        if a == 0:
            x1 = img_width
            y1 = y0
        elif a < np.pi / 2:
            x1 = img_width
            y1 = - np.tan(a) * (x1 - x0) + y0
        elif a == np.pi / 2:
            y1 = 0
            x1 = x0
        elif a < np.pi:
            x1 = 0
            y1 = - np.tan(a) * (x1 - x0) + y0
        elif a == np.pi:
            x1 = 0
            y1 = y0
        elif a < 3 * np.pi / 2:
            x1 = 0
            y1 = - np.tan(a) * (x1 - x0) + y0
        elif a == 3 * np.pi / 2:
            y1 = img_height
            x1 = x0
        elif a < 2 * np.pi:
            x1 = img_width
            y1 = - np.tan(a) * (x1 - x0) + y0
        else:
            raise ValueError('angle `a` error')
        # print('\t x1 {}, y1 {}'.format(x1, y1))
        i, v = siddon_sparse_2d(x0, y0, x1, y1)
        for i_height in range(prj_dims[0]):  # all layers along the z-dim (height) is the same, calculate once
            # ir is the row indices of the sparse matrix, refering to a single detector unit
            ir = i_width + prj_dims[1] * i_height
            # jc is the col indices of the sparse matrix, refering to a single voxel
            jc = i + i_height * img_dims[1] * img_dims[2]  # single x-y layer along the z-dim (height)
            ir = np.repeat(ir, jc.size)  # make `ir` the same size as `jc` and 'v'
            ir_full.append(ir)
            jc_full.append(jc)
            v_full.append(v)

    # use concatenate to combine all the sparse elements
    return np.concatenate(ir_full), np.concatenate(jc_full), np.concatenate(v_full), mat_dims


# defined MACROS
NUM_SUBSET = 1          # number of subset used in OSEM
ELE_PER_SUBSET = 60     # number of angles for each subset, (NUM_SUBSET * ELE_PER_SUBSET == 60)
MAX_ITER = 20           # number of iterations


def main():
    # load projection matrix
    prj_mat = np.fromfile('whole_bone.raw', dtype=np.float32).reshape([60, 128, 128])  # .reshape(-1)

    geometric_angles = np.arange(0, 2 * np.pi, np.pi / 30)
    geometric_distances=[231, 225, 225, 225, 238, 238, 265, 293, 293, 293, 304, 314, 321, 327, 332,
                         334, 335, 335, 335, 335, 335, 314, 294, 293, 290, 286, 281, 281, 269, 269,
                         271, 260, 262, 263, 269, 273, 273, 272, 272, 280, 282, 282, 293, 304, 305,
                         308, 308, 308, 308, 308, 308, 308, 302, 302, 267, 267, 254, 242, 229, 216]

    # save the system matrix of each subset
    sys_mat_list = []
    sys_norm_list = []
    prj_mat_list = []
    for i_subset in range(NUM_SUBSET):
        print('----- SUBSET {} -----'.format(i_subset))
        # select the angle sets
        subset_inds = slice(ELE_PER_SUBSET*i_subset, ELE_PER_SUBSET*(i_subset + 1))
        #subset_inds = slice(i_subset, 60, NUM_SUBSET)
        subset_angles = geometric_angles[subset_inds]
        subset_distances = geometric_distances[subset_inds]
        assert len(subset_angles) == ELE_PER_SUBSET
        irs = []
        jcs = []
        vs = []
        # construct the system matrix for each subset
        for i in range(ELE_PER_SUBSET):
            # construct the system matrix for each angle
            ir, jc, v, size = construct_system_matrix_single_angle(subset_angles[i], subset_distances[i])
            v = v.astype(np.float32)
            ir += i * 128 * 128
            irs.append(ir)
            jcs.append(jc)
            vs.append(v)
        ir = np.concatenate(irs)
        jc = np.concatenate(jcs)
        v = np.concatenate(vs)
        sub_sys_mat = sparse.csr_matrix((v, (ir, jc)), shape=(size[0]* ELE_PER_SUBSET, size[1]))
        sys_mat_list.append(sub_sys_mat)
        # calculate the norm of the system matrix, used for fast computation in iteration
        sys_norm_list.append(np.array(np.sum(sub_sys_mat, axis=0)).reshape(-1))
        # save the sub projection for fast reference
        prj_mat_list.append(prj_mat[subset_inds, :, :].reshape(-1))

    # init the image matrix using all ones
    img_mat = np.ones(shape=[128, 128, 128], dtype=np.float32).reshape(-1)
    for i_iter in range(MAX_ITER):
        print('ITER {}'.format(i_iter))
        for i_subset in range(NUM_SUBSET):
            img_mat = np.multiply(
                np.divide(img_mat, sys_norm_list[i_subset]),
                sys_mat_list[i_subset].transpose() * (prj_mat_list[i_subset] / (sys_mat_list[i_subset] * img_mat)))
            img_mat[np.isnan(img_mat)] = 0

        # save snapshot image
        if np.mod(i_iter, 5) == 0:
            store = img_mat
            store[np.isnan(store)] = 0
            store.tofile('OSEM_recon_iter{}.raw'.format(i_iter))

    # save final results
    store = img_mat
    store[np.isnan(store)] = 0
    store.tofile('OSEM_recon_final.raw')


if __name__ == '__main__':
    main()
