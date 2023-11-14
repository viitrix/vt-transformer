#ifndef _HOST_IMPL_HPP_
#define _HOST_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct HostTensor : public TransformerComputing {
    virtual ~HostTensor() { }
    HostTensor(const ShapeType& shape) {
        if ( _DTYPE_ == DataType::Float ) {
            mem_ = MemoryContext::alloc(shape.numel() * sizeof(float));
        } else if ( _DTYPE_ == DataType::Int ) {
            mem_ = MemoryContext::alloc(shape.numel() * sizeof(int));
        } else if ( _DTYPE_ == DataType::FP16 ) {
            mem_ = MemoryContext::alloc(shape.numel() * sizeof(local_fp16_t));
        } else if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            size_t feature_num = shape.numel() / last_dim;
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128k");
            last_dim += sizeof(float) * 2;
            mem_ = MemoryContext::alloc( feature_num * last_dim );
        } else if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim % Q4_BLOCK_SIZE == 0, "Q4 tensor must has 32 aligened dim");

            size_t numel = shape.numel();
            size_t blk_num = numel / Q4_BLOCK_SIZE;
            mem_ = MemoryContext::alloc(blk_num * sizeof( q4_block_t ));
        } else {
            vt_panic("Can't be here!");
        }
    }
    HostTensor(const ShapeType& shape,  void *mem) : mem_(mem) {
        if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim % Q4_BLOCK_SIZE == 0, "Q4 tensor must has 32 aligened dim");
        }
        if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128k");
        }
    }

    void* data() {
        return mem_;
    }

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;
#ifdef _USING_HPC_OPENMPI_
    ComputingReturn io_mpi_bcast(tensor_t self, int root) override;
    ComputingReturn io_mpi_recv(tensor_t self, int source) override;
    ComputingReturn io_mpi_send(tensor_t self, int dst) override;
#endif
    ComputingReturn io_pipe_read(tensor_t self) override;
    ComputingReturn io_pipe_write(tensor_t self, int n) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_copy(tensor_t self, tensor_t dst) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) override;

protected:
    void* mem_;

    friend struct HostTensor<DataType::Float>;
    friend struct HostTensor<DataType::BF16>;
    friend struct HostTensor<DataType::Int>;
    friend struct HostTensor<DataType::Q8>;
    friend struct HostTensor<DataType::Q4>;
};


} // end of namespace
#endif
