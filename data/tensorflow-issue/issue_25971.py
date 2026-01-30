import numpy as np
import tensorflow as tf

class SparseBinaryMatrixWrapper:
    def __init__(self,Nc,Nt,row,col,batchsize):
        self.Nc=Nc
        self.Nt=Nt
        self.row=row
        self.col=col
        self.batchsize=batchsize
        self.batches=tf.convert_to_tensor(np.r_[0:len(row):batchsize,len(row)])

    @tf.function
    def matvec(self,vec):
        result=tf.zeros(self.Nc,dtype=vec.dtype)
        assert len(vec)==self.Nt
        for i in range(len(self.batches)-1):
            rows=self.row[self.batches[i]:self.batches[i+1]]
            cols=self.col[self.batches[i]:self.batches[i+1]]
            vals=tf.gather(vec,cols) # <-- 1d array
            result=tf.tensor_scatter_nd_add(result,rows[:,None],vals)

        return result

    @tf.function
    def matmul(self,mat):
        result=tf.zeros((self.Nc,mat.shape[1]),dtype=mat.dtype)
        assert mat.shape[0]==self.Nt
        for i in range(len(self.batches)-1):
            rows=self.row[self.batches[i]:self.batches[i+1]]
            cols=self.col[self.batches[i]:self.batches[i+1]]
            vals=tf.gather(mat,cols) # <-- nnz x mat.shape[1]
            result=tf.tensor_scatter_nd_add(result,rows[:,None],vals)

        return result

    @tf.function
    def matTmul(self,mat):
        result=tf.zeros((self.Nt,mat.shape[1]),dtype=mat.dtype)
        assert mat.shape[0]==self.Nc
        for i in range(len(self.batches)-1):
            rows=self.row[self.batches[i]:self.batches[i+1]]
            cols=self.col[self.batches[i]:self.batches[i+1]]
            vals=tf.gather(mat,rows) # <-- nnz x mat.shape[1]
            result=tf.tensor_scatter_nd_add(result,cols[:,None],vals)

        return result


    @tf.function
    def matTvec(self,vec):
        result=tf.zeros(self.Nt,dtype=vec.dtype)
        assert len(vec)==self.Nc
        for i in range(len(self.batches)-1):
            rows=self.row[self.batches[i]:self.batches[i+1]]
            cols=self.col[self.batches[i]:self.batches[i+1]]
            vals=tf.gather(vec,rows)
            result=tf.tensor_scatter_nd_add(result,cols[:,None],vals)
        return result

    @tf.function
    def matTvec(self,vec):
        result=tf.zeros(self.Nt,dtype=vec.dtype)
        assert len(vec)==self.Nc
        for i in range(len(self.batches)-1):
            rows=self.row[self.batches[i]:self.batches[i+1]]
            cols=self.col[self.batches[i]:self.batches[i+1]]
            vals=tf.gather(vec,rows)
            result=tf.tensor_scatter_nd_add(result,cols[:,None],vals)
        return result