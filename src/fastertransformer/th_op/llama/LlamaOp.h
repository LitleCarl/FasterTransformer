/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <random>
#include <sstream>
#include <algorithm>
namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFLlama {
public:
    virtual ~IFLlama() {}
    virtual void forward(th::Tensor&              input_ids,
                         th::Tensor&              input_lengths,
                         th::Tensor&              output_ids,
                         th::Tensor&              sequence_lengths,
                         th::Tensor&              cum_log_probs,
                         const size_t             request_output_len,
                         const size_t             beam_width,
                         th::optional<th::Tensor> top_k_opt,
                         th::optional<th::Tensor> top_p_opt,
                         th::optional<th::Tensor> beam_search_diversity_rate_opt,
                         th::optional<th::Tensor> temperature_opt,
                         th::optional<th::Tensor> len_penalty_opt,
                         th::optional<th::Tensor> repetition_penalty_opt,
                         th::optional<th::Tensor> random_seed_opt,
                         th::optional<int64_t>    return_cum_log_probs_opt) = 0;
};

class MappedFile {
public:
    MappedFile(const std::string& file_path, size_t buffer_size)
        : file_descriptor_(open(file_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR)),
          buffer_size_(buffer_size), buffer_(nullptr) {

        if (file_descriptor_ == -1) {
            throw std::runtime_error("Failed to open the file");
        }

        // Extend the file size to the desired buffer size
        if (lseek(file_descriptor_, buffer_size_ - 1, SEEK_SET) == -1) {
            throw std::runtime_error("Failed to extend the file");
        }
        if (write(file_descriptor_, "", 1) == -1) {
            throw std::runtime_error("Failed to write to the file");
        }

        // Map the file into memory
        buffer_ = mmap(nullptr, buffer_size_, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
        if (buffer_ == MAP_FAILED) {
            throw std::runtime_error("Failed to map the file into memory");
        }
    }

    void Sync() const {
        if (msync(buffer_, buffer_size_, MS_SYNC) == -1) {
            throw std::runtime_error("Failed to sync changes to the file");
        }
    }

    ~MappedFile() {
        if (munmap(buffer_, buffer_size_) == -1) {
            std::cerr << "Failed to unmap the file from memory" << std::endl;
        }
        close(file_descriptor_);
    }

    void* GetBuffer() const {
        return buffer_;
    }

private:
    int file_descriptor_;
    size_t buffer_size_;
    void* buffer_;
};


// namespace uuid {
//     static std::random_device              rd;
//     static std::mt19937                    gen(rd());
//     static std::uniform_int_distribution<> dis(0, 15);
//     static std::uniform_int_distribution<> dis2(8, 11);

//     std::string generate_uuid_v4() {
//         std::stringstream ss;
//         int i;
//         ss << std::hex;
//         for (i = 0; i < 8; i++) {
//             ss << dis(gen);
//         }
//         ss << "-";
//         for (i = 0; i < 4; i++) {
//             ss << dis(gen);
//         }
//         ss << "-4";
//         for (i = 0; i < 3; i++) {
//             ss << dis(gen);
//         }
//         ss << "-";
//         ss << dis2(gen);
//         for (i = 0; i < 3; i++) {
//             ss << dis(gen);
//         }
//         ss << "-";
//         for (i = 0; i < 12; i++) {
//             ss << dis(gen);
//         };
//         return ss.str();
//     }
// }
std::string get_uuid() {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_int_distribution<int> dist(0, 15);

    const char *v = "0123456789abcdef";
    const bool dash[] = { 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 };

    std::string res;
    for (int i = 0; i < 16; i++) {
        if (dash[i]) res += "-";
        res += v[dist(rng)];
        res += v[dist(rng)];
    }
    return res;
}

template<typename T>
class FTLlama: public IFLlama {
public:
    FTLlama(const size_t             head_num,
            const size_t             size_per_head,
            const size_t             inter_size,
            const size_t             layer_num,
            const size_t             vocab_size,
            const size_t             rotary_embedding_dim,
            const float              layernorm_eps,
            const int                start_id,
            const int                end_id,
            const int64_t            tensor_para_size,
            const int64_t            pipeline_para_size,
            const size_t             max_seq_len,
            const bool               use_gptj_residual,
            const vector<th::Tensor> weights, 
            const int64_t  int8_mode):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        rotary_embedding_dim_(rotary_embedding_dim),
        layernorm_eps_(layernorm_eps),
        start_id_(start_id),
        end_id_(end_id),
        use_gptj_residual_(use_gptj_residual),
        weights_(weights),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        int8_mode_(int8_mode)
    {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG, "");
        cublas_wrapper_mutex_ = new std::mutex();

        ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

        Llama_weights_.resizeLayer(layer_num_);

        int hidden_units = head_num_ * size_per_head_; 

        vector<th::Tensor> captured_weights;
        captured_weights.reserve(weights_.size());

        for (int i = 0; i < (int)layer_num_; i++) {
            Llama_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 1 * layer_num_]);

            Llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias = 
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.bias = 
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.beta = 
                get_ptr<T>(weights_[i + 12 * layer_num_]);
            Llama_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.gamma = 
                get_ptr<T>(weights_[i + 13 * layer_num_]);

            captured_weights.push_back(weights_[i + 0 * layer_num_]);
            captured_weights.push_back(weights_[i + 1 * layer_num_]);
            captured_weights.push_back(weights_[i + 3 * layer_num_]);
            captured_weights.push_back(weights_[i + 5 * layer_num_]);
            captured_weights.push_back(weights_[i + 7 * layer_num_]);
            captured_weights.push_back(weights_[i + 9 * layer_num_]);
            captured_weights.push_back(weights_[i + 11 * layer_num_]);
            captured_weights.push_back(weights_[i + 12 * layer_num_]);
            captured_weights.push_back(weights_[i + 13 * layer_num_]);

            if (int8_mode_ == 0) {
                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel = get_ptr<T>(weights_[i + 2 * layer_num_]);
                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel = get_ptr<T>(weights_[i + 4 * layer_num_]);
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel = get_ptr<T>(weights_[i + 6 * layer_num_]);
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.kernel = get_ptr<T>(weights_[i + 8 * layer_num_]);
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel = get_ptr<T>(weights_[i + 10 * layer_num_]);
                captured_weights.push_back(weights_[i + 2 * layer_num_]);
                captured_weights.push_back(weights_[i + 4 * layer_num_]);
                captured_weights.push_back(weights_[i + 6 * layer_num_]);
                captured_weights.push_back(weights_[i + 8 * layer_num_]);
                captured_weights.push_back(weights_[i + 10 * layer_num_]);
            }

            if (int8_mode_ != 0) {
                // Alloc FFN and Attention int8 weights
                ft::deviceMalloc(&int8_weights_ptr_[i][0], hidden_units * 3 * hidden_units / tensor_para_size_);
                ft::deviceMalloc(&int8_weights_ptr_[i][1], hidden_units / tensor_para_size_ * hidden_units);
                ft::deviceMalloc(&int8_weights_ptr_[i][2], hidden_units * inter_size_ / tensor_para_size_);
                ft::deviceMalloc(&int8_weights_ptr_[i][3], hidden_units * inter_size_ / tensor_para_size_);
                ft::deviceMalloc(&int8_weights_ptr_[i][4], inter_size_ / tensor_para_size_ * hidden_units);
                if (int8_mode_ == 1) {
                    // Alloc scales for weight only quant for attention and FFN weights
                    ft::deviceMalloc(&weight_only_scale_ptr_[i][0], 3 * hidden_units / tensor_para_size_);
                    ft::deviceMalloc(&weight_only_scale_ptr_[i][1], hidden_units);
                    ft::deviceMalloc(&weight_only_scale_ptr_[i][2], inter_size_ / tensor_para_size_);
                    ft::deviceMalloc(&weight_only_scale_ptr_[i][3], inter_size_ / tensor_para_size_);
                    ft::deviceMalloc(&weight_only_scale_ptr_[i][4], hidden_units);
                }
                // TODO:cjx 
                ft::FtCudaDataType dtype = ft::FtCudaDataType::FP16;

                auto tensor = weights_[i + 2 * layer_num_].to(torch::kCPU);
                ft::loadWeightFromBufferAndQuantizeForWeightOnly<T>(int8_weights_ptr_[i][0],
                                                        weight_only_scale_ptr_[i][0],
                                                        {(size_t)hidden_units, (size_t)(3 * hidden_units / tensor_para_size_)},
                                                        get_ptr<T>(tensor),
                                                        dtype);
                
                tensor = weights_[i + 4 * layer_num_].to(torch::kCPU);
                ft::loadWeightFromBufferAndQuantizeForWeightOnly<T>(int8_weights_ptr_[i][1],
                                                     weight_only_scale_ptr_[i][1],
                                                     {(size_t)(hidden_units / tensor_para_size_), (size_t)hidden_units},
                                                     get_ptr<T>(tensor),
                                                     dtype);
            
                tensor = weights_[i + 6 * layer_num_].to(torch::kCPU);
                ft::loadWeightFromBufferAndQuantizeForWeightOnly<T>(int8_weights_ptr_[i][2],
                                                     weight_only_scale_ptr_[i][2],
                                                    {(size_t)hidden_units, (size_t)(inter_size_ / tensor_para_size_)},
                                                     get_ptr<T>(tensor),
                                                     dtype);
            
                tensor = weights_[i + 8 * layer_num_].to(torch::kCPU); 
                ft::loadWeightFromBufferAndQuantizeForWeightOnly<T>(int8_weights_ptr_[i][3],
                                                     weight_only_scale_ptr_[i][3],
                                                    {(size_t)hidden_units, (size_t)(inter_size_ / tensor_para_size_)},
                                                     get_ptr<T>(tensor),
                                                     dtype);
                tensor = weights_[i + 10 * layer_num_].to(torch::kCPU);
                ft::loadWeightFromBufferAndQuantizeForWeightOnly<T>(int8_weights_ptr_[i][4],
                                                     weight_only_scale_ptr_[i][4],
                                                    {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units},
                                                     get_ptr<T>(tensor),
                                                     dtype);
                
                
                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.int8_kernel = int8_weights_ptr_[i][0]; 
                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.int8_kernel = int8_weights_ptr_[i][1]; 
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.int8_kernel = int8_weights_ptr_[i][2];
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.int8_kernel = int8_weights_ptr_[i][3];
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.int8_kernel = int8_weights_ptr_[i][4];


                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.weight_only_quant_scale = weight_only_scale_ptr_[i][0];
                Llama_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.weight_only_quant_scale = weight_only_scale_ptr_[i][1];
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.weight_only_quant_scale = weight_only_scale_ptr_[i][2];
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight2.weight_only_quant_scale = weight_only_scale_ptr_[i][3];
                Llama_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.weight_only_quant_scale = weight_only_scale_ptr_[i][4];
            }
        }


        Llama_weights_.pre_decoder_embedding_table   = get_ptr<T>(weights_[14 * layer_num_ + 0]);
        Llama_weights_.post_decoder_layernorm.beta   = get_ptr<T>(weights_[14 * layer_num_ + 1]);
        Llama_weights_.post_decoder_layernorm.gamma  = get_ptr<T>(weights_[14 * layer_num_ + 2]);
        Llama_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[14 * layer_num_ + 3]);
        captured_weights.push_back(weights_[14 * layer_num_]);
        captured_weights.push_back(weights_[14 * layer_num_ + 1]);
        captured_weights.push_back(weights_[14 * layer_num_ + 2]);
        captured_weights.push_back(weights_[14 * layer_num_ + 3]);

        Llama_weights_.setMaxSeqLen(max_seq_len);

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
        weights_ = captured_weights;
    }

    ~FTLlama() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr_.size(); i++) {
                for (auto& x : int8_weights_ptr_[i]) {
                    if (x != nullptr) {
                        ft::deviceFree(x);
                    }
                }
            }

            if (int8_mode_ == 1) {
                for (int i = 0; i < weight_only_scale_ptr_.size(); i++) {
                    for (auto& x : weight_only_scale_ptr_[i]) {
                        if (x != nullptr) {
                            ft::deviceFree(x);
                        }
                    }
                }
            } 
        }
    }

    void forward(th::Tensor&              input_ids,
                 th::Tensor&              input_lengths,
                 th::Tensor&              output_ids,
                 th::Tensor&              sequence_lengths,
                 th::Tensor&              cum_log_probs,
                 const size_t             request_output_len,
                 const size_t             beam_width,
                 th::optional<th::Tensor> top_k_opt,
                 th::optional<th::Tensor> top_p_opt,
                 th::optional<th::Tensor> beam_search_diversity_rate_opt,
                 th::optional<th::Tensor> temperature_opt,
                 th::optional<th::Tensor> len_penalty_opt,
                 th::optional<th::Tensor> repetition_penalty_opt,
                 th::optional<th::Tensor> random_seed_opt,
                 th::optional<int64_t>    return_cum_log_probs_opt) override
    {
        int return_cum_log_probs = return_cum_log_probs_opt.has_value() ? (int)return_cum_log_probs_opt.value() : 0;

        auto           stream       = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator      = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_length   = (size_t)input_ids.size(1);
        const int    total_output_len   = (int)(max_input_length + request_output_len);

        ft::AttentionType attention_type = ft::getAttentionType<T>(size_per_head_,
                                                                   ft::getSMVersion(),
                                                                   true,   // remove_padding
                                                                   0,      // gpt supports any-seq-length fmha
                                                                   true,   // is_fuse
                                                                   false,  // with_relative_position_bias
                                                                   true);  // causal_mask

        ft::Llama<T> llama = ft::Llama<T>(head_num_,
                                          size_per_head_,
                                          inter_size_,
                                          layer_num_,
                                          vocab_size_,
                                          rotary_embedding_dim_,
                                          layernorm_eps_,
                                          start_id_,
                                          end_id_,
                                          end_id_ + 1,  // p/prompt tuning virtual token start id
                                          ft::PromptLearningType::no_prompt,
                                          use_gptj_residual_,
                                          0.0f,  // beam_search_diversity_rate,
                                          1,     // top_k,
                                          0.0,   // top_p,
                                          0,     // random_seed,
                                          1.0f,  // temperature,
                                          1.0f,  // len_penalty,
                                          1.0f,  // repetition_penalty,
                                          tensor_para_,
                                          pipeline_para_,
                                          stream,
                                          &cublas_wrapper,
                                          &allocator,
                                          false,           // is_free_buffer_after_forward
                                          &prop_,          // cuda_device_prop
                                          attention_type,  // attention_type
					                      int8_mode_,		       // int8 mode
                                          nullptr,         // custom_all_reduce_comm
                                          0);              // enable_custom_all_reduce

        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"output_seq_len",
             ft::Tensor{
                 ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};
        if (beam_width > 1 && beam_search_diversity_rate_opt.has_value()) {
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 convert_tensor<float>(beam_search_diversity_rate_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_p_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_p", convert_tensor<float>(top_p_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_k_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_k", convert_tensor<uint>(top_k_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (temperature_opt.has_value()) {
            input_tensors.insert(
                {"temperature", convert_tensor<float>(temperature_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (len_penalty_opt.has_value()) {
            input_tensors.insert(
                {"len_penalty", convert_tensor<float>(len_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (repetition_penalty_opt.has_value()) {
            input_tensors.insert({"repetition_penalty",
                                  convert_tensor<float>(repetition_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (random_seed_opt.has_value()) {
            input_tensors.insert(
                {"random_seed",
                 convert_tensor<unsigned long long int>(random_seed_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            llama.forward(&output_tensors, &input_tensors, &Llama_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
	catch(const std::exception& e) {
        	std::cout << "Caught exception \"" << e.what() << "\"\n";
    	}
//catch (...) {
  //        std::cout << "Runtime error";
    //       exit(-1);
      //  }
    }

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t rotary_embedding_dim_;
    const float  layernorm_eps_;
    const int    start_id_;
    const int    end_id_;
    const bool   use_gptj_residual_;

    // const ft::gptVariantParams Llama_variant_params_;

    std::vector<th::Tensor> weights_;
    cublasLtHandle_t        cublasltHandle_;
    std::mutex*             cublas_wrapper_mutex_;
    ft::cublasAlgoMap*      cublas_algo_map_;
    struct cudaDeviceProp   prop_;
    ft::LlamaWeight<T>      Llama_weights_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    int64_t tensor_para_size_;
    int64_t pipeline_para_size_;
    int64_t int8_mode_ = 0;

    
    std::vector<std::vector<int8_t *>> int8_weights_ptr_ = std::vector<std::vector<int8_t *>> (32, std::vector<int8_t*>(5, nullptr));
    std::vector<std::vector<T *>>      weight_only_scale_ptr_ = std::vector<std::vector<T *>> (32, std::vector<T*>(5, nullptr)); 
};

class LlamaOp: public th::jit::CustomClassHolder {
public:
    LlamaOp(const int64_t            head_num,
            const int64_t            size_per_head,
            const int64_t            inter_size,
            const int64_t            layer_num,
            const int64_t            vocab_size,
            const int64_t            rotary_embedding_dim,
            const double             layernorm_eps,
            const int64_t            start_id,
            const int64_t            end_id,
            const int64_t            tensor_para_size,
            const int64_t            pipeline_para_size,
            const int64_t            max_seq_len,
            const bool               use_gptj_residual,
            const vector<th::Tensor> weights,
            const int64_t int8_mode);

    ~LlamaOp();

    vector<th::Tensor> forward(th::Tensor               input_ids,
                               th::Tensor               input_lengths,
                               const int64_t            output_len,
                               th::optional<int64_t>    beam_width_opt,
                               th::optional<th::Tensor> top_k_opt,
                               th::optional<th::Tensor> top_p_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor> temperature_opt,
                               th::optional<th::Tensor> len_penalty_opt,
                               th::optional<th::Tensor> repetition_penalty_opt,
                               th::optional<th::Tensor> random_seed_opt,
                               th::optional<int64_t>    return_cum_log_probs_opt);

private:
    const at::ScalarType    st_;
    IFLlama*                ftllama;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
