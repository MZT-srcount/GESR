import pycuda.driver as cuda
import pycuda.driver
from pycuda.compiler import SourceModule
import numpy as np

import PyGP


def host_to_gpu(ptr, bytes):
    ptr_gpu = cuda.mem_alloc(bytes)
    cuda.memcpy_htod(ptr_gpu, ptr)
    return ptr_gpu


class MemManager:
    def __init__(self, input_mem_size, cash_mem_size):
        cuda.init()
        self.input_mem_size = input_mem_size
        self.cash_mem_size = cash_mem_size
        self.const_gpu = None
        self.exp_gpu = None
        self.cash_gpu = None
        self.output_gpu = None
        self.fitness_gpu = None
        # self.mem2dcpy = None
        self.initposi = None
        self.np_array = []
        self.mem2dcpy = [cuda.Memcpy2D() for i in range(3)]

    def __del__(self):
        if self.const_gpu is not None:
            del self.const_gpu
        if self.exp_gpu is not None:
            del self.exp_gpu
        if self.cash_gpu is not None:
            del self.cash_gpu
        if self.output_gpu is not None:
            del self.output_gpu
        if self.fitness_gpu is not None:
            del self.fitness_gpu
            
    def cash_clear(self):
        self.cash_gpu = None

    def cash_alloc(self, height):
        if self.cash_gpu is None:
            if self.width_ is None:
                raise ValueError("Please init input memory first")
            if self.width_ * height * PyGP.DATA_TYPE > self.cash_mem_size:
                raise ValueError("cash memory allocation out of limitation: ", self.width_,
                                 (self.width_ * height * PyGP.DATA_TYPE) / (1024 ** 3),
                                 self.cash_mem_size / (1024 ** 3))
            # print('width, height ', self.width_, height)
            (self.cash_gpu, self.cash_pitch) = cuda.mem_alloc_pitch(self.width_ * PyGP.DATA_TYPE, height,
                                                                    PyGP.DATA_TYPE)
        return (self.cash_gpu, self.cash_pitch)

    def get_cash_attr(self):
        return (self.cash_gpu, self.cash_pitch)

    def input_alloc(self, width, height, width_max):
        if width == -1:
            if width_max is None:
                raise ValueError("Please indicate the width_max")
            width = width_max
            while width * height * PyGP.DATA_TYPE > self.input_mem_size:
                width >>= 1
            # width >>= 1
            self.width_ = width
            self.input_height = height
        else:
            if self.input_height > height:
                return (self.width_, self.input_gpu, self.input_pitch)
            else:  # 如果input_height过小还是可能会出问题，需要对width也调整
                self.input_height = height
                del self.input_gpu  # [ ] 释放内存，是否有效？

        if self.width_ * self.input_height * PyGP.DATA_TYPE > self.input_mem_size:
            raise ValueError("The request input memory is out of limitation")
        (self.input_gpu, self.input_pitch) = cuda.mem_alloc_pitch(self.width_ * PyGP.DATA_TYPE, height, PyGP.DATA_TYPE)
        # print(self.width_ * 4, height, self.input_pitch)
        return (self.width_, self.input_gpu, self.input_pitch)

    def get_inputgpu(self):
        return (self.input_gpu, self.input_pitch)

    def const_alloc(self, size):
        if size == 0:
            self.const_gpu = (cuda.mem_alloc(PyGP.DATA_TYPE), PyGP.DATA_TYPE)
        elif self.const_gpu is None:
            self.const_gpu = (cuda.mem_alloc(size), size)
        elif self.const_gpu[1] < size:
            del self.const_gpu
            self.const_gpu = (cuda.mem_alloc(size), size)
        return self.const_gpu[0]

    def output_alloc(self, size):
        if self.output_gpu is None:
            self.output_gpu = (cuda.mem_alloc(size), size)
        elif self.output_gpu[1] < size:
            self.output_gpu = (cuda.mem_alloc(size), size)
        return self.output_gpu[0]

    def fitness_alloc(self, size):
        if self.fitness_gpu is None:
            self.fitness_gpu = (cuda.mem_alloc(size), size)
        elif self.fitness_gpu[1] < size:
            self.fitness_gpu = (cuda.mem_alloc(size), size)
        return self.fitness_gpu[0]

    def exp_alloc(self, size):
        if self.exp_gpu is None:
            self.exp_gpu = (cuda.mem_alloc(size), size)
        if self.exp_gpu[1] < size:
            del self.exp_gpu
            self.exp_gpu = (cuda.mem_alloc(size), size)
        return self.exp_gpu[0]

    def initposi_alloc(self, size):
        if self.initposi is None:
            self.initposi = (cuda.mem_alloc(size), size)
        if self.initposi[1] < size:
            del self.initposi
            self.initposi = (cuda.mem_alloc(size), size)
        return self.initposi[0]

    def host2device(self, host_ptr, device_ptr):
        cuda.memcpy_htod(device_ptr, host_ptr)

    def memcopy_2D(self, dst, dst_pitch, src, src_pitch, width, height, stream, id, src_y_offset=None,
                   dst_y_offset=None):
        # if self.mem2dcpy is None:
        #     self.mem2dcpy = cuda.Memcpy2D()
        mem2dcpy = cuda.Memcpy2D()
        if src_y_offset is not None:
            # print('self.mem2dcpy.src_y: ', self.mem2dcpy.src_y, src_y_offset)
            mem2dcpy.src_y = src_y_offset
        else:
            mem2dcpy.src_y = 0
        if dst_y_offset is not None:
            mem2dcpy.dst_y = dst_y_offset
        else:
            mem2dcpy.dst_y = 0
        if isinstance(src, np.ndarray):
            mem2dcpy.set_src_host(src)
        elif isinstance(src, cuda.DeviceAllocation):
            mem2dcpy.set_src_device(src)
        # print(len(src), np.array(src, dtype = np.float32), len(np.array(src, dtype = np.float32)))
        # mem2dcpy.src_x_in_bytes = i * self.subdataset_size * self.n_terms
        if isinstance(dst, cuda.DeviceAllocation):
            mem2dcpy.set_dst_device(dst)
        elif isinstance(dst, np.ndarray):
            mem2dcpy.set_dst_host(dst)
        mem2dcpy.src_pitch = src_pitch
        mem2dcpy.dst_pitch = dst_pitch
        mem2dcpy.width_in_bytes = width
        mem2dcpy.height = mem2dcpy.src_height = height
        # print(mem2dcpy.src_pitch, mem2dcpy.src_height)
        mem2dcpy(stream)


class Info:
    batch_num = 0
    dataset_size = 0
    input_max = 0
    pop_size = 0
    buffer_id = 0
    buffer_size = 0
    buffer_posi = 0
    iter_id = 0

    def __init__(self, batch_num, subdata_size, input_max, pop_size, data_size):
        self.batch_num = batch_num
        self.subdata_size = subdata_size
        self.data_size = data_size
        self.input_max = input_max
        self.pop_size = pop_size
        self.buffer_id = Info.buffer_id
        self.buffer_size = Info.buffer_size
        self.buffer_posi = Info.buffer_posi

    def set_buffer(self, buffer):
        self.buffer_id = buffer[0]
        self.buffer_size = buffer[1]
        self.buffer_posi = buffer[2]

    def set_iterid(self, iter_id):
        self.iter_id = iter_id

    def get_tuple(self):
        return (self.batch_num, self.subdata_size, self.input_max, self.pop_size, self.buffer_id, self.buffer_size,
                self.buffer_posi, self.iter_id, self.data_size)


class EvalParaInfo:
    def __init__(self, subdataset_size, dataset_size):
        self.subdataset_size = subdataset_size
        self.dataset_size = dataset_size

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def set_offset(self, input_offset, output_offset):
        self.input_offset = input_offset
        self.output_offset = output_offset

    def get_tuple(self):
        return (self.subdataset_size, self.dataset_size, self.batch_idx, self.input_offset, self.output_offset)


mod = SourceModule("""
/************************************************************************* 
 *description: 
 *TODO: 
 *return {*}
 *************************************************************************/
struct EvalParaInfo{
    int subdataset_size, dataset_size;
    int batch_idx;
};
__global__ void evaluation_GPU(double* input, double* output, double* fitness, size_t pitch, int* info){
    int tid = threadIdx.x;
    int time = info[0] / blockDim.x;
    extern __shared__ double fit_shared[];
    fit_shared[tid] = 0;

    double* origin_output = input + info[4];
    if(info[2] == 0 && threadIdx.x == 0){
        fitness[blockIdx.x] = 0;
    }
    __threadfence();
    __syncthreads();

    int pop_init = blockIdx.x * pitch;
    //printf("-=-==-%d\\n", pop_init + info[3]);
    double* output_posi = (double*)((char*)output + pop_init + info[3]);
    for(int i = 0; i < time; ++i){
        int init_posi = i * blockDim.x;

        fit_shared[tid] += (origin_output[init_posi + tid] - output_posi[init_posi + tid]) * (origin_output[init_posi + tid] - output_posi[init_posi + tid]);
        //if(threadIdx.x == 1){
        //    printf("output_posi[init_posi + tid]: %f\\n", output_posi[init_posi + tid]);
        //}
        __threadfence();
    __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if(tid < info[0] % blockDim.x){
        fit_shared[tid] += (origin_output[time * blockDim.x + tid] - output_posi[time * blockDim.x + tid]) * (origin_output[time * blockDim.x + tid] - output_posi[time * blockDim.x + tid]);
    }
    __threadfence();
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            fit_shared[tid] += fit_shared[offset + tid];
        }
        __threadfence();
    __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if(threadIdx.x == 0){
        fitness[blockIdx.x] = sqrt(fit_shared[0] / info[1]);
        //if(blockIdx.x == 0){
        //    printf("%f:%f:%f:%f:%d \\n", fitness[blockIdx.x], fit_shared[0], origin_output[0 + tid], output_posi[0 + tid], info[1]);
        //}
    }

}

__device__ void reduce_sum(double* input, int tid, int size){
    for(int i = size / 2; i > 0; i >>= 1){
        if(tid < size / 2){
            input[tid] += input[tid + i];
        }
        __syncthreads();
    } 
    if(size % 2 == 1 && threadIdx.x == 0){
        input[0] += input[size - 1];
    }
    __syncthreads();

}

__global__ void pearson_rlt_GPU(double* input, double* output, double* results, size_t pitch, int* info, double* XYZ){

    double* X = (double*)((char*)XYZ + blockIdx.x * info[1] * 3 * 8);

    double* Y = (double*)((char*)XYZ + blockIdx.x * info[1] * 3 * 8 + info[1] * 8);
    double* Z = (double*)((char*)XYZ + blockIdx.x * info[1] * 3 * 8 + info[1] * 8 * 2);
    int tid = threadIdx.x;
    //=======================================获取均值
    int pop_init = blockIdx.x * pitch;
    double* output_posi = (double*)((char*)output + pop_init + info[3]);
    while(tid < info[1]){
        X[tid] = output_posi[tid];
        tid += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    tid = threadIdx.x;
    reduce_sum(X, tid, info[1]);

    __threadfence();
    __syncthreads();

    double x_mean = X[0] / (double)info[1];

    tid = threadIdx.x;
    while(tid < info[1]){
        Y[tid] = input[tid];
        tid += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    tid = threadIdx.x;
    reduce_sum(Y, tid, info[1]);

    __threadfence();
    __syncthreads();

    double y_mean = Y[0] / (double)info[1];

    __threadfence();
    __syncthreads();
    //======================================差值
    tid = threadIdx.x;
    while(tid < info[1]){
        X[tid] = output_posi[tid] - x_mean;
        tid += blockDim.x;
    }

    tid = threadIdx.x;
    while(tid < info[1]){
        Y[tid] = input[tid] - y_mean;
        tid += blockDim.x;
    }

    __threadfence();
    __syncthreads();
    tid = threadIdx.x;
    while(tid < info[1]){
        Z[tid] = X[tid] * Y[tid];
        tid += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    tid = threadIdx.x;
    while(tid < info[1]){
        X[tid] *= X[tid];
        Y[tid] *= Y[tid];
        tid += blockDim.x;
    }

    __threadfence();
    __syncthreads();

    tid = threadIdx.x;
    reduce_sum(X, tid, info[1]);
    tid = threadIdx.x;
    reduce_sum(Y, tid, info[1]);
    tid = threadIdx.x;
    reduce_sum(Z, tid, info[1]);
    __threadfence();
    __syncthreads();

    if(threadIdx.x == 0){
        results[blockIdx.x] = Z[0] / (sqrt(X[0]) * sqrt(Y[0]));
    }

}


__global__ void evaluation_GPU_2(double* input, double* output, double* fitness, size_t pitch, int* info){
    int tid = threadIdx.x;
    int time = info[0] / blockDim.x;
    extern __shared__ double fit_shared[];
    fit_shared[tid] = 0;

    double* origin_output = input + info[4];
    if(info[2] == 0 && threadIdx.x == 0){
        fitness[blockIdx.x] = 0;
    }
    __threadfence();
    __syncthreads();

    int pop_init = blockIdx.x * pitch;
    //printf("-=-==-%d\\n", pop_init + info[3]);
    double* output_posi = (double*)((char*)output + pop_init + info[3]);
    for(int i = 0; i < time; ++i){
        int init_posi = i * blockDim.x;

        fit_shared[tid] += (origin_output[init_posi + tid] - output_posi[init_posi + tid]) * (origin_output[init_posi + tid] - output_posi[init_posi + tid]);
        //if(threadIdx.x == 1){
        //    printf("output_posi[init_posi + tid]: %f\\n", output_posi[init_posi + tid]);
        //}
        __threadfence();
    __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if(tid < info[0] % blockDim.x){
        fit_shared[tid] += (origin_output[time * blockDim.x + tid] - output_posi[time * blockDim.x + tid]) * (origin_output[time * blockDim.x + tid] - output_posi[time * blockDim.x + tid]);
    }
    __threadfence();
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            fit_shared[tid] += fit_shared[offset + tid];
        }
        __threadfence();
    __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if(threadIdx.x == 0){
        fitness[blockIdx.x] += fit_shared[0] / info[1];
        //if(blockIdx.x == 0){
        //    printf("%f:%f:%f:%f:%d \\n", fitness[blockIdx.x], fit_shared[0], origin_output[0 + tid], output_posi[0 + tid], info[1]);
        //}
    }

}

/************************************************************************* 
 *description:
 * batch_num: subdataset is divided into multiple minidatasets, and batch_num is the number of minidatasets
 * dataset_size: the subdataset size
 * input_max: the current max input_max
 *TODO: 
 *return {*}
 *************************************************************************/
 #define NUM_MAX 5
 //#define INT_MAX 2e20
 #define UNLOOP 7
struct Info{
    int batch_num, subdataset_size, input_max, pop_size;
    int buffer_id, buffer_size, buffer_posi, iter_id, data_size;
};

//__device__ void P_PV_mul(){  
//}

//__device__ void VV_mul(){
//}

__global__ void execution_GPU(int* program, int* program_iposi, double* dataset, size_t pitch, int* info, double* const_vals){
    //if(threadIdx.x == 0 && blockIdx.x == 0){
    //    printf("info:%d, %d, %d, %d, %d, %d, %d, %d", info[0], info[1], info[2], info[3], info[4], info[5], info[6], pitch);
    //}
    //__syncthreads();
    //(self.batch_num, self.subdata_size, self.input_max, self.pop_size, self.buffer_id, self.buffer_size, self.buffer_posi, self.iter_id, self.data_size)
    int warp_num = blockDim.x / 32, wid = threadIdx.x / 32;
    int bn_forbatch = gridDim.x / info[0];
    int dprocess_size = info[1] / info[0];//如果不能整除呢，貌似没有考虑
    int unloop = 7; // unloop time
    int thridx_tn = 32;//假设每个minidataset由一个warp进行处理

    //printf(" dataset[%d]: %f,%d,%d, %d, %d, %d\\n", threadIdx.x, dataset[threadIdx.x], blockDim.x, gridDim.x, blockIdx.x % bn_forbatch, warp_num, info[3]);
    __syncthreads();
    if((blockIdx.x % bn_forbatch) * warp_num + wid > info[3]){
        return;
    }
    int prog_iid = program_iposi[(blockIdx.x % bn_forbatch) * warp_num + wid];

    int data_iposi = (blockIdx.x / bn_forbatch) * dprocess_size + threadIdx.x % 32;
    char* ds_gpu = (char*)dataset;

    int* prog_iop = program + prog_iid;
    int buffer_posi = info[4] * info[5];

    int op_idx = 0;
    // extern __shared__ float* input_posi[];//[input_max + 1];
    // extern __shared__ float input_data[];//[input_max * unloop];
    double* input_posi[NUM_MAX + 1];
    double input_data[NUM_MAX * UNLOOP];
    double const_data[NUM_MAX];
    while(prog_iop[op_idx] != -1){
        int data_posi = data_iposi;
        int input_size = prog_iop[op_idx + 1];
        for(int i = 0; i < input_size + 1; ++i){
            int input_idx = op_idx + 2 + i;
            if(prog_iop[input_idx] < 0){
                continue;
            }
            if(info[6] + info[5] > prog_iop[input_idx] && prog_iop[input_idx] >= info[6]){
                input_posi[i] = (double*)(ds_gpu + (prog_iop[input_idx] + buffer_posi) * pitch);//位于缓冲位置，需要进行偏移
            }
            else{
                input_posi[i] = (double*)(ds_gpu + prog_iop[input_idx] * pitch);
            }
        }
        int interval = unloop * thridx_tn;
        int data_eposi = dprocess_size * (blockIdx.x / bn_forbatch + 1);
        if(data_eposi > info[1]){
            data_eposi = info[1];
        }

        __threadfence();
        __syncthreads();
        switch(prog_iop[op_idx]){
            case 0://'+'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    if(prog_iop[op_idx + 3] < 0){//两变量均是const
                        const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                        double const_res = const_data[0] + const_data[1];
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_res;
                        }
                    }
                    else{
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_data[0] + input_posi[1][i];
                        }
                    }
                }
                else if(prog_iop[op_idx + 3] < 0){//另一边为const
                    const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = const_data[1] + input_posi[0][i];
                    }
                }
                else{

                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[2 * i] = input_posi[0][input_idx];
                            input_data[2 * i + 1] = input_posi[1][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[2][input_idx] = input_data[2 * i] + input_data[2 * i + 1];
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = input_posi[0][i] + input_posi[1][i];
                    }
                }
                break;
            case 1: //'-'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    if(prog_iop[op_idx + 3] < 0){//两变量均是const
                        const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                        double const_res = const_data[0] - const_data[1];
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_res;
                        }
                    }
                    else{
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_data[0] - input_posi[1][i];
                        }
                    }
                }
                else if(prog_iop[op_idx + 3] < 0){//另一边为const
                    const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = input_posi[0][i] -  const_data[1];
                    }
                }
                else{
                     while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[2 * i] = input_posi[0][input_idx];
                            input_data[2 * i + 1] = input_posi[1][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[2][input_idx] = input_data[2 * i] - input_data[2 * i + 1];
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = input_posi[0][i] - input_posi[1][i];
                    }
                }
                break;
            case 2: //'*'
                if(prog_iop[op_idx + 2] < 0){//是否存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    if(prog_iop[op_idx + 3] < 0){//两变量均是const
                        const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                        double const_res = const_data[0] * const_data[1];
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_res;
                        }
                    }
                    else{
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_data[0] * input_posi[1][i];
                        }
                    }
                }
                else if(prog_iop[op_idx + 3] < 0){//另一边为const
                    const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = const_data[1] * input_posi[0][i];
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[2 * i] = input_posi[0][input_idx];
                            input_data[2 * i + 1] = input_posi[1][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[2][input_idx] = input_data[2 * i] * input_data[2 * i + 1];
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[2][i] = input_posi[0][i] * input_posi[1][i];
                    }
                }
                break;
            case 3: //'/'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    if(prog_iop[op_idx + 3] < 0){//两变量均是const
                        const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                        double const_res = const_data[0];
                        if(fabs(const_data[1]) == 0.0){
                            const_res = const_data[0];
                        }
                        else{
                            const_res = const_data[0] / const_data[1];
                        }

                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = const_res;
                        }
                    }
                    else{
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            if(fabs(input_posi[1][i]) == 0.0){
                                input_posi[2][i] = const_data[0];
                            }
                            else{
                                input_posi[2][i] = const_data[0] / input_posi[1][i];
                            }
                        }
                    }
                }
                else if(prog_iop[op_idx + 3] < 0){//另一边为const
                    const_data[1] = const_vals[-prog_iop[op_idx + 3] - 1];
                    if(fabs(const_data[1]) == 0.0){
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = input_posi[0][i];
                        }
                    }
                    else{
                        for(int i = data_posi; i < data_eposi; i += thridx_tn){
                            input_posi[2][i] = input_posi[0][i] / const_data[1];
                        }
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[2 * i] = input_posi[0][input_idx];
                            input_data[2 * i + 1] = input_posi[1][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            if(fabs(input_data[2 * i + 1]) == 0.0){
                                input_posi[2][input_idx] = input_data[2 * i];
                            }
                            else{
                                input_posi[2][input_idx] = input_data[2 * i] / input_data[2 * i + 1];
                            }
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        if(fabs(input_posi[1][i]) == 0.0){
                            input_posi[2][i] = input_posi[0][i];
                        }
                        else{
                            input_posi[2][i] = input_posi[0][i] / input_posi[1][i];
                        }
                    }
                }
                break;
            case 4://'sin'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = sin(const_data[0]);
                    }
                }
                else{

                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[1][input_idx] = sin(input_data[i]);
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = sin(input_posi[0][i]);
                    }
                }
                break;
            case 5://'cos'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = cos(const_data[0]);
                    }
                }
                else{

                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[1][input_idx] = cos(input_data[i]);
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = cos(input_posi[0][i]);
                    }
                }
                break;
            case 6://'loge'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        if (fabs(const_data[0]) < 1e-12){
                            input_posi[1][i] = log(1e-12);
                        }
                        else{
                            input_posi[1][i] = log(fabs(const_data[0]));
                        }
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            if (fabs(input_data[i]) < 1e-12){
                                input_posi[1][input_idx] = log(1e-12);
                            }
                            else{
                                input_posi[1][input_idx] = log(fabs(input_data[i]));
                            }
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        if (fabs(input_posi[0][i]) < 1e-12){
                            input_posi[1][i] = log(1e-12);
                        }
                        else{
                            input_posi[1][i] = log(fabs(input_posi[0][i]));
                        }
                    }
                }
                break;
            case 7://'exp'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        if (const_data[0] > 8){
                            input_posi[1][i] = exp(8.);
                        }
                        else{
                            input_posi[1][i] = exp(const_data[0]);
                        }
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            if(input_data[i] > 8){
                                input_posi[1][input_idx] = exp(8.);
                            }
                            else{
                                input_posi[1][input_idx] = exp(input_data[i]);
                            }
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        if(input_posi[0][i] > 8){
                            input_posi[1][i] = exp(8.);
                        }
                        else{
                            input_posi[1][i] = exp(input_posi[0][i]);
                        }
                    }
                }
                break;
            case 8://'sqrt'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = sqrt(fabs(const_data[0]));
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[1][input_idx] = sqrt(fabs(input_data[i]));
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = sqrt(fabs(input_posi[0][i]));
                    }
                }
                break;
            case 9://'fabs'
                if(prog_iop[op_idx + 2] < 0){//存在const
                    const_data[0] = const_vals[-prog_iop[op_idx + 2] - 1];
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = fabs(const_data[0]);
                    }
                }
                else{
                    while(data_posi + interval - thridx_tn < data_eposi){
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_data[i] = input_posi[0][input_idx];
                        }
                        for(int i = 0; i < unloop; ++i){
                            int input_idx = data_posi + i * thridx_tn;
                            input_posi[1][input_idx] = fabs(input_data[i]);
                        }
                        data_posi += interval;
                    }
                    for(int i = data_posi; i < data_eposi; i += thridx_tn){
                        input_posi[1][i] = fabs(input_posi[0][i]);
                    }
                }
                break;
            default:
                break;
        }
        __threadfence();
        __syncthreads();
        //if(threadIdx.x == 0 && blockIdx.x == 2){
        //    printf("%d-program oper: %d, %d, %d, %d, %d, %f, %f, %f\\n",op_idx, prog_iop[op_idx], prog_iop[op_idx + 1], prog_iop[op_idx + 2], prog_iop[op_idx + 3], prog_iop[op_idx + 4], input_posi[2][0], input_posi[0][0], input_posi[1][0]);
        //    for(int i = 0; i < 10; ++i){
        //        printf("%f ", input_posi[2][i]);
        //    }
        //    printf("\\n");
        //}
        op_idx += 3 + input_size;
    } 

    __threadfence();
    __syncthreads();
}

//每个block一个subpop,不分batch
//[] 将program中需要backpropagation进行标记，然后不让其被父节点覆盖
//[] dataset应该是subdataset 需要修改过来
__global__ void backpropagation(int* program, int* progs_initposi, double* results, double* drvt_results, double* dataset, size_t pitch, int* info, double* const_vals, long exp_num){
    int wsize_perprog = blockDim.x / (info[3] / gridDim.x);
    int subdata_size = info[1];
    int init_data = threadIdx.x % (wsize_perprog);
    int prog_idx = blockIdx.x * (info[3] / gridDim.x) + threadIdx.x / wsize_perprog;
    int op_idx = progs_initposi[prog_idx];
    int res_initposi = info[8] * prog_idx + subdata_size * info[7];
    char* ds_gpu = (char*)dataset;
    double* input_posi[NUM_MAX + 1];
    int buffer_posi = info[4] * info[5];
    int pop_size = info[3];


    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tn = blockDim.x * gridDim.x;
    int data_size = info[8];
    while(tid < exp_num * subdata_size){
        int pop_id = tid / subdata_size;
        drvt_results[pop_id * data_size + tid % subdata_size] = 1;
        tid += tn;
    }
    __threadfence();
    __syncthreads();

    while(program[op_idx] != -1){
        int data_posi = init_data;
        int input_size = program[op_idx + 1];
        for(int i = 0; i < input_size; ++i){
            int input_idx = op_idx + 2 + i;
            if(info[6] + info[5] > program[input_idx] && program[input_idx] >= info[6]){
                input_posi[i] = (double*)(ds_gpu + (program[input_idx] + buffer_posi) * pitch);//位于缓冲位置，需要进行偏移
            }
            else{
                input_posi[i] = (double*)(ds_gpu + program[input_idx] * pitch);
            }
        }
        double const_data[NUM_MAX];
        int input_locate = program[op_idx + 3 + input_size], input_need = 1 - input_locate;//[] 只支持二维这里，后续需要修改
        //if(threadIdx.x == 32 * 2 && blockIdx.x == 11){
        //    float data_0, data_1;
        //    if(program[op_idx + 2] >= 0){
        //        data_0 = input_posi[0][0];
        //    }
        //    else{
        //        data_0 = const_vals[-program[op_idx + 2] - 1];
        //    }
        //    if(program[op_idx + 3] >= 0){
        //        data_1 = input_posi[1][0];
        //    }
        //    else{
        //        data_1 = const_vals[-program[op_idx + 3] - 1];
        //    }
        //    printf("program oper: %d, %d, %d, %d, %d, %f, %f, %f, %d, %d\\n", program[op_idx], program[op_idx + 1], program[op_idx + 2], program[op_idx + 3], program[op_idx + 4], results[res_initposi], data_0, data_1, input_locate, op_idx);
        //}
        switch(program[op_idx]){
            case 0://'+'
                if(input_locate == 0){
                    if(program[op_idx + 3] < 0){
                        const_data[0] = const_vals[-program[op_idx + 3] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] -= const_data[0];
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] -= input_posi[input_need][i];
                        }
                    }
                }
                else{
                    if(program[op_idx + 2] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] -= const_data[0];
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] -= input_posi[input_need][i];
                        }
                    }
                }
                break;
            case 1://'-'
                if(input_locate == 0){
                    if(program[op_idx + 3] < 0){
                        const_data[0] = const_vals[-program[op_idx + 3] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] += const_data[0];
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] += input_posi[input_need][i];
                        }
                    }
                }
                else{
                    if(program[op_idx + 2] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] = const_data[0] - results[res_initposi + i];
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] = input_posi[input_need][i] - results[res_initposi + i];
                        }
                    }
                }
                break;
            case 2://'*'
                if(input_locate == 0){
                    if(program[op_idx + 3] < 0){
                        const_data[0] = const_vals[-program[op_idx + 3] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(const_data[0]) == 0.0){
                                results[res_initposi + i] = NAN;//[] 是否应该这么处理？？不太确定
                            }
                            else{
                                results[res_initposi + i] /= const_data[0];
                            }
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(input_posi[input_need][i]) == 0.0){
                                results[res_initposi + i] = NAN;
                            }
                            else{
                                results[res_initposi + i] /= input_posi[input_need][i];
                            }
                        }   
                    }
                }
                else{
                    if(program[op_idx + 2] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(const_data[0]) == 0.0){
                                results[res_initposi + i] = NAN;//[] 是否应该这么处理？？不太确定
                            }
                            else{
                                results[res_initposi + i] /= const_data[0];
                            }
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(input_posi[input_need][i]) == 0.0){

                                results[res_initposi + i] = NAN;
                            }
                            else{
                                results[res_initposi + i] /= input_posi[input_need][i];
                            }
                        }   
                    }
                }
                break;
            case 3://'/'
                if(input_locate == 0){
                    if(program[op_idx + 3] < 0){
                        const_data[0] = const_vals[-program[op_idx + 3] - 1];
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] *= const_data[0];
                        }
                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            results[res_initposi + i] *= input_posi[input_need][i];
                        }
                    }
                }
                else{
                    if(program[op_idx + 2] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2] - 1];

                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(results[res_initposi + i]) == 0.0){
                                results[res_initposi + i] = NAN;
                            }
                            else{
                                results[res_initposi + i] = const_data[0] / results[res_initposi + i];
                            }
                        }

                    }
                    else{
                        for(int i = data_posi; i < info[1]; i += wsize_perprog){
                            if(fabs(results[res_initposi + i]) == 0.0){
                                results[res_initposi + i] = NAN;
                            }
                            else{
                                results[res_initposi + i] = input_posi[input_need][i] / results[res_initposi + i];
                            }
                        }
                    }
                }
                break;

            case 4://'sin'
                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] > 1 || results[res_initposi + i] < -1){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = asin(results[res_initposi + i]);
                    }
                }
                break;
            case 5://'cos'

                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] > 1 || results[res_initposi + i] < -1){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = acos(results[res_initposi + i]);
                    }
                }
                break;
            case 6://'log'

                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] > 10){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = exp(results[res_initposi + i]);
                    }
                }
                break;
            case 7://'exp'

                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] < 0){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = log(results[res_initposi + i]);
                    }
                }
                break;
            case 8://'sqrt'

                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] < 0){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = (results[res_initposi + i]) * (results[res_initposi + i]);
                    }
                }
                break;
            case 9://'fabs'

                for(int i = data_posi; i < info[1]; i += wsize_perprog){
                    if(results[res_initposi + i] < 0){
                        results[res_initposi + i] = NAN;
                    }
                    else{
                        results[res_initposi + i] = (results[res_initposi + i]);
                    }
                }
                break;
        }

        __threadfence();
        __syncthreads();

        switch(program[op_idx]){
            case 0://'+'
                break;
            case 1://'-'
                if(input_locate == 1){
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= -1;
                    }
                }
                break;
            case 2://'*'
                if(program[op_idx + 2 + input_need] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= const_data[0];
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= input_posi[input_need][i];
                    }
                }
                break;
            case 3://'/'
                if(input_locate == 0){
                    if(program[op_idx + 2 + input_need] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                        if(fabs(const_data[0]) == 0.0){
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                drvt_results[res_initposi + i] *= 1;
                            }
                        }
                        else{
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                drvt_results[res_initposi + i] /= const_data[0];
                            }
                        }
                    }
                    else{
                        for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                            if(fabs(input_posi[input_need][i]) == 0.0){
                                drvt_results[res_initposi + i] *= 1;
                            }
                            else{
                                drvt_results[res_initposi + i] /= input_posi[input_need][i];
                            }
                        }
                    }
                }
                else{
                    if(program[op_idx + 2 + input_need] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                        for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                            if(fabs(results[res_initposi + i]) == 0.0){
                                drvt_results[res_initposi + i] = NAN;
                            }
                            else{
                                drvt_results[res_initposi + i] *= -const_data[0] / (results[res_initposi + i] * results[res_initposi + i]);
                            }
                        }
                    }
                    else{
                        for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                            if(fabs(results[res_initposi + i]) == 0.0){

                                drvt_results[res_initposi + i] = NAN;
                            }
                            else{
                                drvt_results[res_initposi + i] *= -input_posi[input_need][i] / (results[res_initposi + i] * results[res_initposi + i]);
                            }
                        }
                    }
                }

                break;

            case 4://'sin'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= cos(const_data[0]);
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= cos(results[res_initposi + i]);
                    }
                }
                break;
            case 5://'cos'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= -sin(const_data[0]);
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= -sin(results[res_initposi + i]);
                    }
                }
                break;
            case 6://'log'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        if(!(fabs(const_data[0]) < 1e-12)){
                            drvt_results[res_initposi + i] *= fabs(1 / const_data[0]);
                        }
                        else{
                            drvt_results[res_initposi + i] *= fabs(1 / 1e-12);
                        }
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){

                        if(!(fabs(results[res_initposi + i]) < 1e-12)){
                            drvt_results[res_initposi + i] *= fabs(1 / results[res_initposi + i]);
                        }
                        else{
                            drvt_results[res_initposi + i] *= fabs(1 / 1e-12);
                        }
                    }
                }
                break;
            case 7://'exp'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        
                        if (const_data[0] > 8){
                            drvt_results[res_initposi + i] *= exp(8.);
                        }
                        else{
                            drvt_results[res_initposi + i] *= exp(const_data[0]);
                        }
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        
                        if (results[res_initposi + i] > 8){
                            drvt_results[res_initposi + i] *= exp(8.);
                        }
                        else{
                            drvt_results[res_initposi + i] *= exp(results[res_initposi + i]);
                        }
                    }
                }
                break;
            case 8://'sqrt'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                    
                        drvt_results[res_initposi + i] *= 1 / (2.f * sqrt(const_data[0]));
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        drvt_results[res_initposi + i] *= 1 / (2.f * sqrt(results[res_initposi + i]));
                    }
                }
                break;
            case 9://'fabs'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        if (const_data[0] < 0){
                            drvt_results[res_initposi + i] *= -1;
                        }
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        if (results[res_initposi + i] < 0){
                            drvt_results[res_initposi + i] *= -1;
                        }
                    }
                }
                break;
        }

        op_idx += 4 + input_size;
        __threadfence();
        __syncthreads();

        if(program[op_idx] == -1){
            prog_idx += pop_size;
            if(prog_idx >= exp_num){
                break;
            }
            op_idx = progs_initposi[prog_idx];
            res_initposi = info[8] * prog_idx + subdata_size * info[7];
        }
    }
    //    __threadfence();
    //    __syncthreads();
    //if(threadIdx.x == 32 * 2 && blockIdx.x == 11){
    //    printf("program oper: %d, %d, %d, %d, %d, %f\\n", program[op_idx], program[op_idx + 1], program[op_idx + 2], program[op_idx + 3], program[op_idx + 4], results[res_initposi]);
    //    for(int i = 0; i < 10; ++i){
    //        printf("%f ", results[res_initposi + i]);
    //    }
    //    printf("\\n");
    //}
}

__global__ void derivative(int* program, int* progs_initposi, double* results, double* dataset, int pitch, int* info, double* const_vals, int exp_num){
    int wsize_perprog = blockDim.x / (info[3] / gridDim.x);//每个prog分配到的线程数量
    int subdata_size = info[1];
    int init_data = threadIdx.x % (wsize_perprog);
    int prog_idx = blockIdx.x * (info[3] / gridDim.x) + threadIdx.x / wsize_perprog;
    int op_idx = progs_initposi[prog_idx];
    int res_initposi = info[8] * prog_idx + subdata_size * info[7];
    char* ds_gpu = (char*)dataset;
    double* input_posi[NUM_MAX + 1];
    int buffer_posi = info[4] * info[5];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tn = blockDim.x * gridDim.x;
    int data_size = info[8], pop_size = info[3];
    while(tid < exp_num * subdata_size){
        int pop_id = tid / subdata_size;
        results[pop_id * data_size + tid % subdata_size] = 1;
        tid += tn;
    }
    __threadfence();
    __syncthreads();
    while(program[op_idx] != -1){
        int data_posi = init_data;
        int input_size = program[op_idx + 1];
        for(int i = 0; i < input_size; ++i){
            int input_idx = op_idx + 2 + i;
            if(info[6] + info[5] > program[input_idx] && program[input_idx] >= info[6]){
                input_posi[i] = (double*)(ds_gpu + (program[input_idx] + buffer_posi) * pitch);//位于缓冲位置，需要进行偏移
            }
            else{
                input_posi[i] = (double*)(ds_gpu + program[input_idx] * pitch);
            }
        }
        double const_data[NUM_MAX];
        int input_locate = program[op_idx + 3 + input_size], input_need = 1 - input_locate;//[] 只支持二维这里，后续需要修改

        switch(program[op_idx]){
            case 0://'+'
                break;
            case 1://'-'
                if(input_locate == 1){
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= -1;
                    }
                }
                break;
            case 2://'*'
                if(program[op_idx + 2 + input_need] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= const_data[0];
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= input_posi[input_need][i];
                    }
                }
                break;
            case 3://'/'
                if(input_locate == 0){
                    if(program[op_idx + 2 + input_need] < 0){
                        const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                        if(fabs(const_data[0]) == 0.0){
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                results[res_initposi + i] *= 1;
                            }
                        }
                        else{
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                results[res_initposi + i] /= const_data[0];
                            }
                        }
                    }
                    else{
                        for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                            if(fabs(input_posi[input_need][i]) == 0.0){
                                if (fabs(results[res_initposi + i] - input_posi[input_need][i]) <= 1e-12){
                                    results[res_initposi + i] *= 1;
                                }
                                else{
                                    results[res_initposi + i] *= 1;
                                }
                            }
                            else{
                                results[res_initposi + i] /= input_posi[input_need][i];
                            }
                        }
                    }
                }
                else{
                    if(program[op_idx + 2 + input_locate] < 0){
                        const_data[1] = const_vals[-program[op_idx + 2 + input_locate] - 1];

                        if(program[op_idx + 2 + input_need] < 0){
                            const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                            if(const_data[1] * const_data[1] == 0.0){
                                for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                    results[res_initposi + i] *= -const_data[0];
                                }
                            }
                            else{
                                for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                    results[res_initposi + i] *= -const_data[0] / (const_data[1] * const_data[1]);
                                }
                            }
                        }
                        else{
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                if(fabs(const_data[1] * const_data[1]) == 0.0){
                                    results[res_initposi + i] *= -input_posi[input_need][i];
                                }
                                else{
                                    results[res_initposi + i] *= -input_posi[input_need][i] / (const_data[1] * const_data[1]);
                                }
                            }
                        }

                    }
                    else{
                        if(program[op_idx + 2 + input_need] < 0){
                            const_data[0] = const_vals[-program[op_idx + 2 + input_need] - 1];
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                if(fabs(input_posi[input_locate][i] * input_posi[input_locate][i]) == 0.0){
                                    results[res_initposi + i] *= -const_data[0];
                                }
                                else{
                                    results[res_initposi + i] *= -const_data[0] / (input_posi[input_locate][i] * input_posi[input_locate][i]);
                                }
                            }
                        }
                        else{
                            for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                                if(fabs(input_posi[input_locate][i] * input_posi[input_locate][i]) == 0.0){

                                    if (fabs(input_posi[input_need][i] - input_posi[input_locate][i] * input_posi[input_locate][i]) <= 1e-12){
                                        results[res_initposi + i] *= -input_posi[input_need][i];
                                    }
                                    else{
                                        results[res_initposi + i] *= -input_posi[input_need][i];
                                    }
                                }
                                else{
                                    results[res_initposi + i] *= -input_posi[input_need][i] / (input_posi[input_locate][i] * input_posi[input_locate][i]);
                                }
                            }
                        }
                    }
                }

                break;

            case 4://'sin'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= cos(const_data[0]);
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= cos(input_posi[0][i]);
                    }
                }
                break;
            case 5://'cos'

                if(program[op_idx + 2 + input_locate] < 0){
                    const_data[0] = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= -sin(const_data[0]);
                    }
                }
                else{
                    for(int i = data_posi; i < subdata_size; i += wsize_perprog){
                        results[res_initposi + i] *= -sin(input_posi[0][i]);
                    }
                }
                break;
        }

        op_idx += 4 + input_size;
        __threadfence();
        __syncthreads();
        if(program[op_idx] == -1){
            prog_idx += pop_size;
            if(prog_idx >= exp_num){
                break;
            }
            op_idx = progs_initposi[prog_idx];
            res_initposi = info[8] * prog_idx + subdata_size * info[7];
        }
    }

}


__global__ void gradient(int* program, int* progs_initposi, double* results, size_t res_pitch, double* dataset, size_t pitch, int* info, double* const_vals, int exp_num){

    char* ds = (char*)dataset;
    double* res = (double*)((char*)results + res_pitch * blockIdx.x);
    int data_size = info[1];
    int tid = threadIdx.x, t_n = blockDim.x;
    while(tid < data_size){
        res[tid] = 1;
        tid += t_n;
    }
    __syncthreads();
    int thread_num = blockDim.x;
    int op_idx = progs_initposi[blockIdx.x];
    while(program[op_idx] != -1){
        int input_size = program[op_idx + 1];
        int input_locate = program[op_idx + input_size + 2];
        int input_need = op_idx + 3 - input_locate;
        int tid = threadIdx.x;
        switch(program[op_idx]){
            case 0://'+'
                break;
            case 1://'-'
                if(input_locate == 1){
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= -1;
                    }
                }
                break;
            case 2://'*'
                if(program[input_need] < 0){
                    double const_data = const_vals[-program[input_need] - 1];
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= const_data;
                    }
                }
                else{
                    double* input = (double*)(ds + program[input_need] * pitch);
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= input[i];
                    }
                }
                break;
            case 3://'/'
                if(input_locate == 0){
                    if(program[input_need] < 0){
                        double const_data = const_vals[-program[input_need] - 1];
                        if(fabs(const_data) == 0.0){
                            for(int i = tid; i < data_size; i += thread_num){
                                res[i] *= 1;
                            }
                        }
                        else{
                            for(int i = tid; i < data_size; i += thread_num){
                                res[i] /= const_data;
                            }
                        }
                    }
                    else{
                        double* input_1 = (double*)(ds + program[input_need] * pitch);
                        for(int i = tid; i < data_size; i += thread_num){
                            if(fabs(input_1[i]) == 0.0){
                                res[i] *= 1;
                            }
                            else{
                                res[i] /= input_1[i];
                            }
                        }
                    }
                }
                else{

                    if(program[op_idx + 2 + input_locate] < 0){
                        double const_data_1 = const_vals[-program[op_idx + 2 + input_locate] - 1];

                        if(program[input_need] < 0){
                            double const_data_0 = const_vals[-program[input_need] - 1];
                            if(const_data_1 == 0.0){
                                for(int i = tid; i < data_size; i += thread_num){
                                    res[i] *= -const_data_0;
                                }
                            }
                            else{
                                for(int i = tid; i < data_size; i += thread_num){
                                    res[i] *= -const_data_0 / (const_data_1 * const_data_1);
                                }
                            }
                        }
                        else{//````
                            double* input_1 = (double*)(ds + program[input_need] * pitch);
                            if(fabs(const_data_1) == 0.0){
                                for(int i = tid; i < data_size; i += thread_num){
                                    res[i] *= -input_1[i];
                                }
                            }
                            else{
                                for(int i = tid; i < data_size; i += thread_num){
                                    res[i] *= -input_1[i] / (const_data_1 * const_data_1);
                                }
                            }
                        }

                    }
                    else{
                        double* input_1 = (double*)(ds + program[op_idx + 2 + input_locate] * pitch);
                        if(program[input_need] < 0){
                            double const_data_0 = const_vals[-program[input_need] - 1];
                            for(int i = tid; i < data_size; i += thread_num){
                                if(fabs(input_1[i]) == 0.0){
                                    res[i] *= -const_data_0;
                                }
                                else{
                                    res[i] *= -const_data_0 / (input_1[i] * input_1[i]);
                                }
                            }
                        }
                        else{
                            double* input_2 = (double*)(ds + program[input_need] * pitch);
                            for(int i = tid; i < data_size; i += thread_num){
                                if(fabs(input_1[i]) == 0.0){
                                    res[i] *= -input_2[i];
                                }
                                else{
                                    res[i] *= -input_2[i] / (input_1[i] * input_1[i]);
                                }
                            }
                        }
                    }
                }

                break;

            case 4://'sin'

                if(program[op_idx + 2 + input_locate] < 0){
                    double const_data_0 = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= cos(const_data_0);
                    }
                }
                else{
                    double* input = (double*)(ds + program[op_idx + 2 + input_locate] * pitch);
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= cos(input[i]);
                    }
                }
                break;
            case 5://'cos'

                if(program[op_idx + 2 + input_locate] < 0){
                    double const_data_0 = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= -sin(const_data_0);
                    }
                }
                else{
                    double* input = (double*)(ds + program[op_idx + 2 + input_locate] * pitch);
                    for(int i = tid; i < data_size; i += thread_num){
                        res[i] *= -sin(input[i]);
                    }
                }

                break;
            case 6://'log'

                if(program[op_idx + 2 + input_locate] < 0){
                    double const_data_0 = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = tid; i < data_size; i += thread_num){
                        if(!(fabs(const_data_0) < 1e-12)){
                            res[i] *= fabs(1 / const_data_0);
                        }
                        else{
                            res[i] *= fabs(1 / 1e-12);
                        }
                    }
                }
                else{
                    double* input = (double*)(ds + program[op_idx + 2 + input_locate] * pitch);
                    for(int i = tid; i < data_size; i += thread_num){
                        if(!(fabs(input[i]) < 1e-12)){
                            res[i] *= fabs(1 / input[i]);
                        }
                        else{
                            res[i] *= fabs(1 / 1e-12);
                        }
                    }
                }
                break;
            case 7://'exp'

                if(program[op_idx + 2 + input_locate] < 0){
                    double const_data_0 = const_vals[-program[op_idx + 2 + input_locate] - 1];
                    for(int i = tid; i < data_size; i += thread_num){
                        
                        if (const_data_0 > 8){
                            res[i] *= exp(8.);
                        }
                        else{
                            res[i] *= exp(const_data_0);
                        }
                    }
                }
                else{
                    double* input = (double*)(ds + program[op_idx + 2 + input_locate] * pitch);
                    for(int i = tid; i < data_size; i += thread_num){
                        
                        if (input[i] > 8){
                            res[i] *= exp(8.);
                        }
                        else{
                            res[i] *= exp(input[i]);
                        }
                    }
                }
                break;
        }
        op_idx += 4 + program[op_idx + 1];
    }
}
""")

