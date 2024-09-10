/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "handle.h"
#include "definitions.h"
#include "logging.h"
#include "status.h"
#include "utility.hpp"

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
ROCSPARSELT_KERNEL void init_kernel(){};

void _rocsparselt_handle::init()
{
    // Layer mode
    log_bench = false;
    char* str_layer_mode;
    if((str_layer_mode = getenv("HIPSPARSELT_LOG_LEVEL")) == NULL)
    {
        layer_mode = rocsparselt_layer_mode_none;
        if((str_layer_mode = getenv("HIPSPARSELT_LOG_MASK")) != NULL)
        {
            layer_mode = strtol(str_layer_mode, nullptr, 0);
        }
    }
    else
    {
        layer_mode = rocsparselt_layer_mode_none;
        switch(atoi(str_layer_mode))
        {
        case rocsparselt_layer_level_log_dump_tensor:
            layer_mode |= rocsparselt_layer_mode_log_dump_tensor;
        case rocsparselt_layer_level_log_api:
            layer_mode |= rocsparselt_layer_mode_log_api;
        case rocsparselt_layer_level_log_info:
            layer_mode |= rocsparselt_layer_mode_log_info;
        case rocsparselt_layer_level_log_hints:
            layer_mode |= rocsparselt_layer_mode_log_hints;
        case rocsparselt_layer_level_log_trace:
            layer_mode |= rocsparselt_layer_mode_log_trace;
        case rocsparselt_layer_level_log_error:
            layer_mode |= rocsparselt_layer_mode_log_error;
            break;
        default:
            layer_mode = rocsparselt_layer_mode_none;
            break;
        }
    }

    if((str_layer_mode = getenv("HIPSPARSELT_LOG_BENCH")) != NULL)
    {
        log_bench = (atoi(str_layer_mode) > 0);
    }

    // Open log file
    if(layer_mode & 0xff)
    {
        log_trace_ofs = new std::ofstream();
        open_log_stream(&log_trace_os, log_trace_ofs, "HIPSPARSELT_LOG_FILE");
    }

    // Open log_bench file
    if(log_bench)
    {
        log_bench_ofs = new std::ofstream();
        open_log_stream(&log_bench_os, log_bench_ofs, "HIPSPARSELT_LOG_BENCH_FILE");
    }

    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    log_trace(this, "handle::init", "hipGetDevice");

    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));
    log_trace(this, "handle::init", "hipGetDeviceProperties", device);

    // Device wavefront size
    wavefront_size = properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

    is_init = (uintptr_t)(this);

    alg_selections = std::make_shared<std::vector<rocsparselt_matmul_alg_selection*>>();
}

void _rocsparselt_handle::destroy()
{
    is_init = 0;
    // Close log files
    if(log_trace_ofs)
    {
        if(log_trace_ofs->is_open())
            log_trace_ofs->close();
        delete log_trace_ofs;
        log_trace_ofs = nullptr;
    }
    if(log_bench_ofs)
    {
        if(log_bench_ofs->is_open())
            log_bench_ofs->close();
        delete log_bench_ofs;
        log_bench_ofs = nullptr;
    }
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_mat_descr& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", format=" << rocsparselt_matrix_type_to_string(t.m_type)
           << ", row=" << t.m << ", col=" << t.n << ", ld=" << t.ld << ", alignment=" << t.alignment
           << ", datatype=" << hipDataType_to_string(t.type)
           << ", order=" << rocsparselt_order_to_string(t.order);
    if(t.m_type == rocsparselt_matrix_type_structured)
        stream << ", sparsity=" << rocsparselt_sparsity_to_string(t.sparsity);

    stream << ", batchSize=" << t.num_batches << ", batchStride=" << t.batch_stride << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_descr& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", opA=" << rocsparselt_operation_to_string(t.op_A)
           << ", opB=" << rocsparselt_operation_to_string(t.op_B) << ", matA=" << *(t.matrix_A)
           << ", matB=" << *(t.matrix_B) << ", matC=" << *(t.matrix_C);
    if(t.matrix_C != t.matrix_D)
        stream << ", matD=" << *(t.matrix_D);
    stream << ", computeType=" << rocsparselt_compute_type_to_string(t.compute_type)
           << ", activation=" << rocsparselt_activation_type_to_string(t.activation)
           << ", activation_relu_upperbound=" << t.activation_relu_upperbound
           << ", activation_relu_threshold=" << t.activation_relu_threshold
           << ", activation_leakyrelu_alpha=" << t.activation_leakyrelu_alpha
           << ", activation_tanh_alpha=" << t.activation_tanh_alpha
           << ", activation_tanh_beta=" << t.activation_tanh_beta
           << ", activation_gelu_scaling=" << t.activation_gelu_scaling
           << ", bias_pointer=" << t.bias_pointer << ", bias_stride=" << t.bias_stride
           << ", bias_type=" << hipDataType_to_string(t.bias_type) << ", m=" << t.m << ", n=" << t.n
           << ", k=" << t.k << ", is_sparse_a=" << t.is_sparse_a << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_alg_selection& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", alg=" << t.alg << ", config_id=" << t.config_id
           << ", config_max_id=" << t.config_max_id << ", search_iterations=" << t.search_iterations
           << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_matmul_plan& t)
{
    stream << "{"
           << "ptr=" << (&t) << ", matmul=" << *(t.matmul_descr)
           << ", alg_selection=" << *(t.alg_selection) << "}";
    return stream;
}

template<typename T>
std::ostream& dump_tensor_data(std::ostream& stream, const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    constexpr bool is_int = std::is_same<T, int8_t>();
    using Tp              = std::conditional_t<is_int, int32_t, float>;
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    stream << "---------- " << name << "(" << n1 << "x" << n2 << "," << n3 <<"," << s1<<"," << s2<<"," << s3 << ") ----------\n";
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                stream << static_cast<Tp>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]) << "|";
            }
            stream << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            stream << "\n";
    }
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const _rocsparselt_tensor_data& t)
{
    switch(t.type)
    {
    case HIP_R_32F:
       return dump_tensor_data<float>(stream, t.text, reinterpret_cast<float*>(t.data), t.m, t.n, t.num_batches, t.stride0, t.stride1, t.batch_stride);
    case HIP_R_16BF:
       return dump_tensor_data<hip_bfloat16>(stream, t.text, static_cast<hip_bfloat16*>(t.data), t.m, t.n, t.num_batches, t.stride0, t.stride1, t.batch_stride);
    case HIP_R_16F:
       return dump_tensor_data<half>(stream, t.text, static_cast<half*>(t.data), t.m, t.n, t.num_batches, t.stride0, t.stride1, t.batch_stride);
    case HIP_R_8I:
       return dump_tensor_data<int8_t>(stream, t.text, static_cast<int8_t*>(t.data), t.m, t.n, t.num_batches, t.stride0, t.stride1, t.batch_stride);
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E5M2_FNUZ:
    default:
        return stream;
    }
}
