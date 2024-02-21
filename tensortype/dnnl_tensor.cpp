#include "dnnl_tensor.hpp"

namespace vt {

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_dump(tensor_t self) {
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
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> DNNLTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        return (size_t) self->items() * sizeof(float);
    }
    if ( _DTYPE_ == DataType::Int ) {
        return (size_t) self->items() * sizeof(int);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return (size_t) self->items() * sizeof(local_fp16_t);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

    const char* d = (const char *)data();
    size_t len = std::get<1>(self->op_sizeof(self));
    wf.write(d, len);
    wf.close();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
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

template <DataType _DTYPE_>
ComputingReturn op_conv2d(tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) {
    return OP_TODO_ERROR;
}

tensor_t create_dnnl_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::Float>* tensor = new DNNLTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::FP16>* tensor = new DNNLTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::Int>* tensor = new DNNLTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
