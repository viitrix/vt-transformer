#ifndef _TT_ENGINE_HPP_
#define _TT_ENGINE_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"
#include "host_tensor.hpp"

#ifdef _USING_DEVICE_DNNL_
#include "dnnl_tensor.hpp"
#endif

#ifdef _USING_DEVICE_CUDA_
#include "cuda_tensor.hpp"
#endif

#ifdef _USING_DEVICE_CUDA_
#include "dcu_tensor.hpp"
#endif

#include "dag.hpp"

#endif
