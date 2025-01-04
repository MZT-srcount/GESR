import PyGP.util
import numpy as np
import pycuda.driver as cuda
from PyGP import Program
import pycuda.autoinit as autoinit


from src.cuda_backend import MemManager, mod, Info
class Execution:
    def __call__(self, progs, dataset):
        return self.execute_gpu(progs, dataset)

    def execute_gpu(self, progs: list, dataset):
        if len(progs) == 0:
            return
        if not isinstance(progs[0], Program):
            raise ValueError("input must be transferred to Program type first")
        n_terms = progs[0].n_terms
        funcs_set = progs[0].funcs
        if len(dataset) != n_terms:
            raise ValueError("Please input the entire dataset")

        progs_size = len(progs)
        dataset_len = len(dataset[0])
        encoder = PyGP.util.Encoder()
        exp_attr = encoder(progs)  # (e_clts, e_iposi, id_altr, cvals)
        gpu_height = exp_attr[2][0]
        gpu_width = dataset_len
        cuda_manager = MemManager(4 * (1024 ** 3), 0)
        (subdataset_size, input_gpu, input_pitch) = cuda_manager.input_alloc(-1, gpu_height, gpu_width)

        exps_gpu = cuda_manager.exp_alloc(len(exp_attr[0]) * 4)
        initposi_gpu = cuda_manager.initposi_alloc(len(exp_attr[1]) * 4)
        const_vals_gpu = cuda_manager.const_alloc(len(exp_attr[3]) * PyGP.DATA_TYPE)
        cuda_manager.host2device(np.array(exp_attr[3], dtype=np.float32 if PyGP.DATA_TYPE == 4 else np.float64),
                                 const_vals_gpu)
        cuda_manager.host2device(np.array(exp_attr[0], dtype=np.int32), exps_gpu)
        cuda_manager.host2device(np.array(exp_attr[1], dtype=np.int32), initposi_gpu)

        # iter_time = int(dataset_len / subdataset_size) 这里必须单次解决，因此input_alloc还有点问题
        info = Info(1, subdataset_size, funcs_set.max_arity(), len(progs), dataset_len)
        execution_GPU = mod.get_function("execution_GPU")
        t_dataset = PyGP.dataset_transform(1, dataset, subdataset_size)

        stream = cuda.Stream()
        cuda_manager.memcopy_2D(input_gpu, input_pitch,
                                t_dataset[0][0], subdataset_size * PyGP.DATA_TYPE,
                                subdataset_size * PyGP.DATA_TYPE, n_terms, stream, 0)
        # print(self.n_terms * self.subdataset_size, self.n_terms, self.subdataset_size)
        execution_GPU(exps_gpu, initposi_gpu, input_gpu, np.int64(input_pitch),
                      cuda.In(np.array(info.get_tuple(), dtype=np.int32)), const_vals_gpu,
                      block=(int(32), 1, 1),
                      grid=(int((progs_size) * PyGP.BATCH_NUM), 1, 1),
                      stream=stream)

        output = np.empty(subdataset_size * progs_size).astype(np.float32 if PyGP.DATA_TYPE == 4 else np.float64)
        cuda_manager.memcopy_2D(output,
                                subdataset_size * PyGP.DATA_TYPE,
                                input_gpu, input_pitch,
                                subdataset_size * PyGP.DATA_TYPE, progs_size,
                                cuda.Stream(), 0,
                                src_y_offset=int(n_terms + 1))

        ## evaluation
        # fit_cpu = []
        # output = output.reshape(progs_size, -1)
        # print(output.shape)
        # for i in range(progs_size):
        #     fit_cpu.append(np.sqrt(np.dot(output[i] - fitness, output[i] - fitness) / len(fitness)))
        self.exec_para = {
            'exps':exps_gpu,
            'exp_posi':initposi_gpu,
            'input_gpu':input_gpu,
            'input_pitch':input_pitch,
            'info':cuda.In(np.array(info.get_tuple(), dtype=np.int32)),
            'cvals':const_vals_gpu
        }
        self.expms = exp_attr[4]
        self.cuda_manager = cuda_manager
        autoinit.context.synchronize()
        # print('output: ', output)

        return output # （输出）
