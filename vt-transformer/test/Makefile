# Makefile — build vt-transformer tests.
#
# 用法：
#   make                # 构建全部测试到 ./bin/
#   make check          # 构建 + 运行会自行终止的单元测试
#   make <name>         # 只构建一个，例如 make vt_endpoint_test
#   make clean          # 删除 ./bin/
#
# 二进制产出在 ./bin/<test_name>，依赖文件 .d 也落在那里。

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra -MMD -MP
INCLUDES  := -I.. -I../opt

BIN := bin

# 终止型单元测试：自己跑完就退出，可以串到 `make check` 里。
UNIT_TESTS := vt_test vt_cache_test vt_radix_test safetensors_test vt_backend_test

# 长跑 / 集成型：`make all` 会构建，但 `make check` 不跑（不会自行退出）。
LOOP_TARGETS := vt_endpoint_test

ALL_TARGETS := $(UNIT_TESTS) $(LOOP_TARGETS)

.PHONY: all check clean $(ALL_TARGETS)
all: $(addprefix $(BIN)/, $(ALL_TARGETS))

check: $(addprefix $(BIN)/, $(UNIT_TESTS))
	@set -e; for t in $(UNIT_TESTS); do \
		echo "--- $$t ---"; \
		./$(BIN)/$$t; \
	done

clean:
	rm -rf $(BIN)

$(BIN):
	mkdir -p $(BIN)

# 便捷别名：`make vt_test` 等价于 `make bin/vt_test`。
$(ALL_TARGETS): %: $(BIN)/% ;

# 模式规则：单 .cpp、无额外源文件或库。
$(BIN)/vt_cache_test $(BIN)/vt_radix_test $(BIN)/safetensors_test: $(BIN)/%: %.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# vt_test 还需要链接 ../core/vt.cpp。
$(BIN)/vt_test: vt_test.cpp ../core/vt.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) vt_test.cpp ../core/vt.cpp -o $@

# vt_endpoint_test 链接 libzmq。
$(BIN)/vt_endpoint_test: vt_endpoint_test.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ -lzmq -lpthread

# vt_backend_test：自终止型集成测试，跑全 pipeline（UserMsg → forward → DetokenizeMsg）。
# 链接 vt_backend.cpp + vt_scheduler.cpp + vt.cpp + libzmq。
$(BIN)/vt_backend_test: vt_backend_test.cpp \
                        ../core/vt_backend.cpp ../core/vt_scheduler.cpp ../core/vt.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ -lzmq -lpthread

-include $(BIN)/*.d
