## 如何移植一个全新的模型

首先查看 VT-Transformer 中的算子集，是否满足这个模型的所有必要计算，如果满足计算能力，找到最接近的模型作为原型进行移植操作。

如果算子集没有满足，则需要在 tensortype/computing.hpp 增加算子顶，并且在 tensortype.cpp、nn_operators.cpp文件增加调用入口，DAG 函数入口等。
最后还需要目标的硬件平台上，编写真正的计算实现函数。
