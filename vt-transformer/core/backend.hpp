#ifndef _VT_BACKEND_HPP_
#define _VT_BACKEND_HPP_

#include <vector>

#include "request.hpp"

namespace vt {

// 后端接口基类: 抽象出 "从前端拉取请求" 的统一接口。
// 参数语义 (毫秒):
//   timeOut == 0 : 非阻塞, 只做一次检查;
//   timeOut >  0 : 最多等待 timeOut 毫秒;
//   timeOut <  0 : 无限期等待, 直到有请求到达。
template <typename Token = int32_t>
struct BackendBase {
    BackendBase() = default;
    virtual ~BackendBase() = default;
    BackendBase(const BackendBase&)            = delete;
    BackendBase& operator=(const BackendBase&) = delete;

    virtual void tryFetchRequest(std::vector<Request<Token>>& newRequest, int timeOut) = 0;
};

} // end of namespace
#endif
