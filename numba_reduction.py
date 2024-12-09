import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestReduction(CUDATestCase):
    """
    Test shared memory reduction
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def ex_reduction(self, insize):
        import numpy as np
        from numba import cuda
        from numba.types import int32

        # reduction.allocate.begin
        # generate data
        ndata = insize
        nthreads = 256
        shrsize = nthreads*2
        a = cuda.to_device(np.arange(ndata))
        nelem = len(a)
        nblocks = (nelem - 1) // nthreads + 1

        # allocating the output array
        d_psum = cuda.to_device(np.zeros(nblocks, dtype=np.int32))
        @cuda.jit
        def array_sum(data, psum):
            tid = cuda.threadIdx.x
            i = cuda.grid(1)
            length = len(data)

            #we take in the shared size
            shared_buffer = cuda.shared.array(shrsize, int32)

            # Load data or zero if out of bounds
            val = data[i] if i < length else 0
            shared_buffer[tid] = val

            cuda.syncthreads()

            # Now perform reductiion
            step = 1
            while step < cuda.blockDim.x:
                if (tid % (2 * step)) == 0:
                    shared_buffer[tid] += shared_buffer[tid + step]
                step *= 2
                cuda.syncthreads()

            if tid == 0:
                psum[cuda.blockIdx.x] = shared_buffer[0]

        array_sum[nblocks, nthreads](a, d_psum)
        host_b = d_psum.copy_to_host()

        # Validating results for each block
        for block_idx in range(nblocks):
            block_start = block_idx * nthreads
            block_end = min((block_idx + 1) * nthreads, nelem)
            expected = np.sum(np.arange(block_start, block_end, dtype=np.int32))
            np.testing.assert_equal(host_b[block_idx], expected)

    #unit testcases which also includes input sizes which not multiples of thread block size to make the code more generalizable
    def test_reduction_256(self):
        self.ex_reduction(256)

    def test_reduction_1024(self):
        self.ex_reduction(1024)

    def test_reduction_10240(self):
        self.ex_reduction(10240)

    def test_reduction_20480(self):
        self.ex_reduction(20480)

    def test_reduction_12345(self):
        self.ex_reduction(12345)

    def test_reduction_10241(self):
       self.ex_reduction(10241)

    def test_reduction_9987(self):
        self.ex_reduction(9987)

if __name__ == "__main__":
    unittest.main()
