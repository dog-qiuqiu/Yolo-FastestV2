#include <iostream>
#include <fstream>

#include "musicyolo.h"
#include "NvOnnxParser.h"
#include <device_launch_parameters.h>


using namespace nvinfer1;

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

Engine::Engine(const Options &options)
    : m_options(options) {
    if (!m_options.doesSupportDynamicBatchSize) {
        std::cout << "Model does not support dynamic batch size, using optBatchSize and maxBatchSize of 1" << std::endl;
        m_options.optBatchSize = 1;
        m_options.maxBatchSize = 1;
    }
}

bool Engine::build(std::string onnxModelPath) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set the max supported batch size
    builder->setMaxBatchSize(m_options.maxBatchSize);

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    const auto input = network->getInput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputL = inputDims.d[1];

    // Specify the optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims2(1, inputL));
    optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims2(m_options.optBatchSize, inputL));
    optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims2(m_options.maxBatchSize, inputL));
    config->addOptimizationProfile(optProfile);

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.precision == Precision::FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;                                                                                                                                                                                            
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

Engine::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    if (!m_engine->bindingIsInput(0)) {
        throw std::runtime_error("Error, the model does not have an input!");
    }
    auto dims = m_engine->getBindingDimensions(0);
    m_inputL = dims.d[1];

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Allocate the input and output buffers
    m_buffers.resize(m_engine->getNbBindings());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate memory for the input
    // Allocate enough to fit the max batch size (we could end up using less later)
    checkCudaErrorCode(cudaMallocAsync(&m_buffers[0], m_options.maxBatchSize * dims.d[1] * sizeof(float), stream));

    // Allocate buffers for the outputs
    m_outputLengthsFloat.clear();
    for (int i = 1; i < m_engine->getNbBindings(); ++i) {
        if (m_engine->bindingIsInput(i)) {
            // This code implementation currently only supports models with a single input
            throw std::runtime_error("Implementation currently only supports models with single input");
        }

        uint32_t outputLenFloat = 1;
        auto outputDims = m_engine->getBindingDimensions(i);
        for (int j = 1; j < outputDims.nbDims; ++j) {
            // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
            outputLenFloat *= outputDims.d[j];
        }

        m_outputLengthsFloat.push_back(outputLenFloat);
        // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
        checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

void Engine::checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

bool Engine::runInference(const std::vector<cv::cuda::GpuMat> &inputs, std::vector<std::vector<std::vector<float>>>& featureVectors) {
    // First we do some error checking
    if (inputs.empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    if (!m_options.doesSupportDynamicBatchSize) {
        if (inputs.size() > 1) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Model does not support running batch inference!" << std::endl;
            std::cout << "Please only provide a single input" << std::endl;
            return false;
        }
    }

    auto inputDimsOriginal = m_engine->getBindingDimensions(0);
    auto batchSize = static_cast<int32_t>(inputs.size());

    auto& input = inputs[0]; // 取出batch中的第一个数据
    if (input.cols != inputDimsOriginal.d[1]) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Input does not have correct size!" << std::endl;
        std::cout << "Execpted: (" << inputDimsOriginal.d[1] << ")"  << std::endl;
        std::cout << "Got: (" << input.channels() << ")" << std::endl;
        std::cout << "Ensure you resize your input audio to the correct size" << std::endl;
        return false;
    }

    nvinfer1::Dims2 inputDims = {batchSize, inputDimsOriginal.d[1]};
    m_context->setBindingDimensions(0, inputDims); // Define the batch size

    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Copy over the input data and perform the preprocessing
    cv::cuda::GpuMat gpu_dst(1, input.rows * input.cols * inputs.size() , CV_16SC1);
    size_t width = input.cols * input.rows;
    for (size_t idx=0; idx< inputs.size(); idx++) {
        std::vector<cv::cuda::GpuMat> input_channels;
        for (int c = 0; c < input.channels(); ++c)
        {
            input_channels.emplace_back(
                cv::cuda::GpuMat(input.rows, input.cols, CV_16SC1, &(gpu_dst.ptr()[width * c + width * input.channels() * idx]))
            );

        };
        cv::cuda::split(inputs[idx], input_channels);  // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    gpu_dst.convertTo(mfloat, CV_32FC1, 1.f / 32768.f);

    auto *dataPointer = mfloat.ptr<void>();

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    checkCudaErrorCode(cudaMemcpyAsync(m_buffers[0], dataPointer, mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float), cudaMemcpyDeviceToDevice, inferenceCudaStream));

    // Run inference.
    bool status = m_context->enqueueV2(m_buffers.data(), inferenceCudaStream, nullptr);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        for (int outputBinding = 1; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
            // First binding is the input which is why we start at index 1
            std::vector<float> output;
            auto outputLenFloat = m_outputLengthsFloat[outputBinding - 1];
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

std::string Engine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    engineName+= "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);
    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t newDim, const cv::Scalar &bgcolor) {
    float r = std::min(newDim / (input.cols * 1.0), newDim / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(newDim, newDim, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}