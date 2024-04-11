#include "host_tensor.hpp"


namespace vt {

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> HostTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        return (size_t) self->items() * sizeof(float);
    }
    if ( _DTYPE_ == DataType::Int ) {
        return (size_t) self->items() * sizeof(int);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return (size_t) self->items() * sizeof(local_fp16_t);
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        auto shape = self->shape();
        size_t last_dim = shape.vec().back();
        size_t feature_num = self->items() / last_dim;
        last_dim += sizeof(float) * 2;
        return last_dim * feature_num;
    }
    if ( _DTYPE_ == DataType::Q4 ) {
        auto shape = self->shape();
        size_t numel = shape.numel();
        size_t blk_num = numel / Q4_BLOCK_SIZE;
        return blk_num * sizeof( q4_block_t );
    }
    if ( _DTYPE_ == DataType::PQ ) {
        size_t items = self->items();
        size_t ret = sizeof(local_fp16_t) * 64 * 2 * PQ_S_  + items * 3 / 8;
        vt_assert( ret == size_ , " double check PQ's memory size");
        return ret;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::op_zero(tensor_t self) {
    size_t s = std::get<1>( self->op_sizeof(self) );
    memset( mem_, 0, s );
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::op_copy(tensor_t self, tensor_t src) {
    if ( src->is_host() ) {
        size_t size = std::get<1>( self->op_sizeof(self) );
        memcpy(data(), src->device_data(), size );
        return OP_OK;
    }

#ifdef _USING_DEVICE_CUDA_
    if ( src->is_cuda() ) {
        auto stream = ComputingContext::cuda_stream;
        size_t size = std::get<1>( self->op_sizeof(self) );
        CUDA_CHECK(cudaMemcpyAsync(data(), src->device_data(), size , cudaMemcpyDeviceToHost, stream));
        return OP_OK;
    }
#endif

#ifdef _USING_DEVICE_DCU_
    if ( src->is_dcu() ) {
        auto stream = ComputingContext::dcu_stream;
        size_t size = std::get<1>( self->op_sizeof(self) );
        HIP_CHECK(hipMemcpyAsync(data(), src->device_data(), size , hipMemcpyDeviceToHost, stream));
        return OP_OK;
    }
#endif

#ifdef _USING_DEVICE_COREX_
    if ( src->is_corex() ) {
        auto stream = ComputingContext::corex_stream;
        size_t size = std::get<1>( self->op_sizeof(self) );
        COREX_CHECK(cudaMemcpyAsync(data(), src->device_data(), size , cudaMemcpyDeviceToHost, stream));
        return OP_OK;
    }
#endif

#ifdef _USING_DEVICE_DNNL_
    if ( src->is_dnnl() ) {
        size_t s = std::get<1>( self->op_sizeof(self) );
        void* x = src->device_data();
        void* y = data();
        memcpy(y, x, s);
        return OP_OK;
    }
#endif

    vt_panic("Can't be here!");
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::op_embed(tensor_t self, tensor_t table, tensor_t output) {
    const size_t feature_size = table->shape().vec().back();
    const int items = self->items();
    int* tokens = (int * )self->host_int()->data();

    std::string odev = output->device_name();
    if(  ! ( odev == "dnnl" || odev == "acl" ) ) {
        vt_panic("Can't do op_embed on output's device!");
    }

    if ( table->dtype() == DataType::Float && output->dtype() == DataType::Float ) {
        float* dst = (float *)output->device_data();
        float* src = (float *)table->host_float()->data();
        for ( int i = 0; i < items; i++) {
            float* emb = &src[ tokens[i] * feature_size ];
            memcpy(dst, emb, sizeof(float)*feature_size);
            dst += feature_size;
        }
        return OP_OK;
    }
    if ( table->dtype() == DataType::FP16 && output->dtype() == DataType::FP16) {
        local_fp16_t* dst = (local_fp16_t *)output->device_data();
        local_fp16_t* src = (local_fp16_t *)table->host_fp16()->data();
        for ( int i = 0; i < items; i++) {
            local_fp16_t* emb = &src[ tokens[i] * feature_size ];
            memcpy(dst, emb, sizeof(local_fp16_t)*feature_size);
            dst += feature_size;
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> HostTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new HostTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Int ) {
        ShapeType newShape(newShape_);
        int *newData = (int *)data() + offset;
        auto* newCpuTensor = new HostTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        ShapeType newShape(newShape_);
        local_fp16_t *newData = (local_fp16_t *)data() + offset;
        auto* newCpuTensor = new HostTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        auto shape = self->shape();
        size_t last_dim = shape.vec().back();
        vt_assert(offset % last_dim == 0, "Q8's view must aligen with last dim");
        vt_assert(newShape_.back() == last_dim, "Q8's view must aligen with last dim");

        ShapeType newShape(newShape_);
        void *newData = (char *)data() + (offset / last_dim) * ( last_dim + 2 * sizeof(float) );
        auto* newCpuTensor = new HostTensor<DataType::Q8>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Q4 ) {
        ShapeType newShape(newShape_);
        vt_assert(offset % Q4_BLOCK_SIZE == 0, "Q4's view must aligen with Q4_BLOCK_T");
        void *newData = (char *)data() + (offset / Q4_BLOCK_SIZE) * sizeof(q4_block_t);
        auto* newCpuTensor = new HostTensor<DataType::Q4>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    if ( _DTYPE_ == DataType::Float ) {
        float* d = (float *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (float *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        int* d = (int *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (int *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16_t* d = (local_fp16_t *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;
        d = (local_fp16_t *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        size_t last_dim = self->shape().vec().back();
        size_t feature_num = self->items() / last_dim;

        q8_head_t* target = (q8_head_t *)data();
        std::cout << "First " << first8 << " : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Last " << first8 << " : ";
        target = (q8_head_t *)( (char *)data() + (feature_num - 1) * (last_dim + 2 * sizeof(float)));
        for (size_t i = last_dim - 8; i < last_dim; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q4 ) {
        size_t block_num = self->items() / Q4_BLOCK_SIZE;

        q4_block_t* d = (q4_block_t *)data();
        std::cout << "First " << first8 << " : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q4(d, i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Last " << first8 << " : ";
        d += block_num - 1;
        for (size_t i = Q4_BLOCK_SIZE - 8; i < Q4_BLOCK_SIZE; i++) {
            std::cout << dequantize_q4(d, i) << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::PQ ) {
        const size_t gsize = self->items() / PQ_S_;

        const local_fp16_t*    pqtab = (local_fp16_t *)mem_;
        const uint8_t*  pqidx = (uint8_t *)mem_ + PQ_S_ * 64 * 2 * sizeof(local_fp16_t);

        std::cout << "First 8 : ";
        {
            uint8_t a = pqidx[0];
            uint8_t b = pqidx[1];
            uint8_t c = pqidx[2];

            int i0 = a >> 2;
            int i1 = ( (a & 0x3) << 4)  + (b >> 4);
            int i2 = ( (b & 0x0F) << 2) + (c >> 6);
            int i3 = c & 0x3F;

            std::cout << fp16_to_fp32(pqtab[i0*2 + 0]) << " " << fp16_to_fp32(pqtab[i0*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i1*2 + 0]) << " " << fp16_to_fp32(pqtab[i1*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i2*2 + 0]) << " " << fp16_to_fp32(pqtab[i2*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i3*2 + 0]) << " " << fp16_to_fp32(pqtab[i3*2 + 1]) << " ";
        }
        std::cout << std::endl;

        std::cout << "Last 8 : ";
        {
            int offset = self->items() * 3 / 8 - 3;

            uint8_t a = pqidx[0 + offset];
            uint8_t b = pqidx[1 + offset];
            uint8_t c = pqidx[2 + offset];

            int i0 = a >> 2;
            int i1 = ( (a & 0x3) << 4)  + (b >> 4);
            int i2 = ( (b & 0x0F) << 2) + (c >> 6);
            int i3 = c & 0x3F;

            int s = (self->items() - 1) / gsize;
            pqtab = pqtab + s * 128;

            std::cout << fp16_to_fp32(pqtab[i0*2 + 0]) << " " << fp16_to_fp32(pqtab[i0*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i1*2 + 0]) << " " << fp16_to_fp32(pqtab[i1*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i2*2 + 0]) << " " << fp16_to_fp32(pqtab[i2*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i3*2 + 0]) << " " << fp16_to_fp32(pqtab[i3*2 + 1]) << " ";
        }

        std::cout << std::endl;
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

    const char* d = (const char *)data();
    size_t len = std::get<1>(self->op_sizeof(self));
    wf.write(d, len);
    wf.close();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
    std::ifstream inf(fileName, std::ios::binary);
    if ( ! inf.is_open() ) {
        std::cout << "Can't open " << fileName << std::endl;
        vt_panic("Can't open file");
    }

    if (_DTYPE_ == DataType::Float) {
        size_t ret = inf.read( (char *)data(), sizeof(float) * self->items() ).gcount();
        vt_assert(ret == sizeof(float) * self->items(), "file size dont't match tensor");
    } else  if (_DTYPE_ == DataType::Int) {
        size_t ret = inf.read( (char *)data(), sizeof(int) * self->items() ).gcount();
        vt_assert(ret == sizeof(int) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::FP16) {
        size_t ret = inf.read( (char *)data(), sizeof(local_fp16_t) * self->items() ).gcount();
        vt_assert(ret == sizeof(local_fp16_t) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q8 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q4 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::PQ ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)mem_, s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else {
        vt_panic("DataType don't support");
    }

    inf.close();
    return OP_OK;
}

#ifdef _USING_HPC_MPI_
template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_mpi_recv(tensor_t self, int source) {
    if ( _DTYPE_ == DataType::Float ) {
        MPI_Recv(data(), self->items(), MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        MPI_Recv(data(), self->items(), MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        MPI_Recv(data(), self->items(), MPI_SHORT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_mpi_bcast(tensor_t self, int root) {
    if ( _DTYPE_ == DataType::Float ) {
        MPI_Bcast(data(), self->items(), MPI_FLOAT, root, MPI_COMM_WORLD);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        MPI_Bcast(data(), self->items(), MPI_INT, root, MPI_COMM_WORLD);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        MPI_Bcast(data(), self->items(), MPI_SHORT, root, MPI_COMM_WORLD);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_mpi_send(tensor_t self, int dst) {
    if ( _DTYPE_ == DataType::Float ) {
        MPI_Send(data(), self->items(), MPI_FLOAT, dst, 0, MPI_COMM_WORLD);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        MPI_Send(data(), self->items(), MPI_INT, dst, 0, MPI_COMM_WORLD);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        MPI_Send(data(), self->items(), MPI_SHORT, dst, 0, MPI_COMM_WORLD);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}
#endif

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_pipe_read(tensor_t self) {
    auto size = std::get<1>( self->op_sizeof(self) );
    int ret = CollectiveContext::pipe_read(data(), size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn HostTensor<_DTYPE_>::io_pipe_write(tensor_t self, int n) {
    auto size = std::get<1>( self->op_sizeof(self) );
    int ret = CollectiveContext::pipe_write(n, data(), size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    return OP_OK;
}

tensor_t create_host_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Float>* tensor = new HostTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::FP16>* tensor = new HostTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Int>* tensor = new HostTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Q8>* tensor = new HostTensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Q4>* tensor = new HostTensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_pq(std::vector<size_t>& shape_, const int S) {
    ShapeType shape(shape_);
    HostTensor<DataType::PQ>* tensor = new HostTensor<DataType::PQ>(shape, S);
    return std::make_shared<TensorType>(tensor, shape);
}

}
