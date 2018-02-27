import numpy as np


class SparseVectorRegister:
    """
    SparseVectorRegister:
    aims to record all the non-zero elements in a certain vector.
    After the recording process is over, one can call the `close` method to get all the previously registered elements,
    including the index `i` and the value `v`.
    """
    def __init__(self):
        self.i_ = list()
        self.v_ = list()

    def register(self, i, v):
        self.i_.append(i)
        self.v_.append(v)

    def close(self):
        return np.array(self.i_), np.array(self.v_)


class SparseMatrixRegister:
    def __init__(self):
        self.ir_ = list()
        self.jc_ = list()
        self.v_ = list()

    def register(self, ir, jc, v):
        self.ir_.append(ir)
        self.jc_.append(jc)
        self.v_.append(v)

    def register_vector(self, ir, sparse_vector):
        raise NotImplementedError

    def close(self):
        return np.array(self.ir_), np.array(self.jc_), np.array(self.v_)
