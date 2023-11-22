#include "dcu_tensor.hpp"

namespace vt {

tensor_t create_dcu_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Float>* tensor = new DCUTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::FP16>* tensor = new DCUTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Int>* tensor = new DCUTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
