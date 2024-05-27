#ifndef _HOST_IMPL_HPP_
#define _HOST_IMPL_HPP_

#include "vt.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct HostTensor : public TransformerComputing {
    ~HostTensor();
    HostTensor(const ShapeType& shape);

public:
    ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn io_pipe_read(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn io_pipe_write(ComputingContext* ctx, tensor_t self, int dst) override;
    std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) override;

protected:
    const size_t size_;
    const bool owner_;
    void* mem_;

    friend struct HostTensor<DataType::F32>;
    friend struct HostTensor<DataType::I32>;
    friend struct HostTensor<DataType::F16>;
    friend struct HostTensor<DataType::BF16>;
    friend struct HostTensor<DataType::Q8>;
    friend struct HostTensor<DataType::Q4>;
    friend struct HostTensor<DataType::PQ>;
};

} // end of namespace
#endif
