## 如何适配一个新的硬件计算平台

增加一个新的硬件平台，需要完成两部：

* 在 tensortype/tensortype.h(cpp) 两个文件中，增加一个新的编译宏，以及一个新的 TensorType 类型，主要可以参考 DCU 的相关代码
* 增加 对应平台的 计算实现文件，实现 tensortype/computing.hpp 定义的必要函数
* 如果有必要，增加更多的 kernel 实现函数，例如 dcu_kernels 目录下所做的


注意，可以参考完整的实现案例，DCU平台，在每一个使用 _USING_DEVICE_DCU_ 宏的地方，模仿增加即可。

