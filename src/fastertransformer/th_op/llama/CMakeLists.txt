# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
#find_package(pybind11  CONFIG)

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

add_library(th_llama STATIC LlamaOp.cc)
#pybind11_add_module(th_llama STATIC LlamaOp.cc)

set_property(TARGET th_llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_llama PUBLIC "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY} Llama th_utils nccl_utils)
