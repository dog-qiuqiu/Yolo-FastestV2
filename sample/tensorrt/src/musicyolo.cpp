#include <iostream>
#include <fstream>

#include "musicyolo.h"
#include "NvOnnxParser.h"
#include <device_launch_parameters.h>
#include <iomanip>
#include <fstream>
#include <cmath>

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

    inputWidth = 352;
    inputHeight = 192;
    scaleH = 0.458333f;
    scaleW = 0.008523f;
    hop = 176;
    n_fft = 32768;
    confThresh = 0.3;
    nmsThresh = 0.4;
    numAnchor = 3;
    std::vector<float> bias {18.70,98.86, 29.01,101.33, 41.40,98.15,
                             57.29,105.94, 86.45,93.03, 159.47,96.56};
    anchor.assign(bias.begin(), bias.end());

    std::vector<int> dims {15, 12, 22, 6, 11};
    outdims.assign(dims.begin(), dims.end());
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

bool Engine::runInference(const std::vector<int16_t> &input, std::vector<std::vector<std::vector<float>>>& featureVectors) {
    // First we do some error checking
    if (input.empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    // if (!m_options.doesSupportDynamicBatchSize) {
    //     if (inputs.size() > 1) {
    //         std::cout << "===== Error =====" << std::endl;
    //         std::cout << "Model does not support running batch inference!" << std::endl;
    //         std::cout << "Please only provide a single input" << std::endl;
    //         return false;
    //     }
    // }

    auto inputDimsOriginal = m_engine->getBindingDimensions(0);
    // auto batchSize = static_cast<int32_t>(inputs.size());
    auto batchSize = 1;

    if (input.size() != inputDimsOriginal.d[1]) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Input does not have correct size!" << std::endl;
        std::cout << "Execpted: (" << inputDimsOriginal.d[1] << ")"  << std::endl;
        std::cout << "Got: (" << input.size() << ")" << std::endl;
        std::cout << "Ensure you resize your input audio to the correct size" << std::endl;
        return false;
    }

    nvinfer1::Dims2 inputDims = {batchSize, inputDimsOriginal.d[1]};
    m_context->setBindingDimensions(0, inputDims); // Define the batch size

    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }    

    std::vector<float> inputF(input.size());
    std::transform(input.begin(), input.end(), inputF.begin(), [](const int16_t& val) -> float {
                return static_cast<float>(val) / 32768.f; });

    // std::ofstream out;
    // out.open("cqt.bin", std::ios::out | std::ios::binary);
    // out.write(reinterpret_cast<const char*>(inputF.data()), inputF.size() * sizeof(float));
    // out.close();
    // std::cout << "audio ==== " << inputF.size() <<  std::endl;

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    checkCudaErrorCode(cudaMemcpyAsync(m_buffers[0], inputF.data(), inputF.size() * sizeof(float), cudaMemcpyHostToDevice, inferenceCudaStream));

    // Run inference.
    bool status = m_context->enqueueV2(m_buffers.data(), inferenceCudaStream, nullptr);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
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

//特征图后处理
int Engine::predHandle(const std::vector<std::vector<float>> &out, std::vector<TargetBox> &dstBoxes, 
                       const float scaleW, const float scaleH, const float thresh)
{    //do result
    for (int i = 0; i < out.size(); i++) {   
        int stride;
        int outW, outH, outC;

        outH = outdims[2 * i + 1];
        outW = outdims[2 * i + 2];
        outC = outdims[0];
        
        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; ++h) {
            for (int w = 0; w < outW; ++w) {
                int off = w * outC + h * outW * outC;
                for (int b = 0; b < numAnchor; ++b) {    
                    TargetBox tmpBox;
                    int offset = b * 4 + off;
                    float score = out[i][off + 4 * numAnchor + b];
                    if (score > thresh) {
                        float bcx, bcy, bw, bh;

                        bcx = ((out[i][offset + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((out[i][offset + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((out[i][offset + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((out[i][offset + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];
                        
                        tmpBox.x1 = bcx - 0.5 * bw;
                        tmpBox.y1 = bcy - 0.5 * bh;
                        tmpBox.x2 = bcx + 0.5 * bw;
                        tmpBox.y2 = bcy + 0.5 * bh;
                        tmpBox.score = score;

                        dstBoxes.push_back(tmpBox);
                    }
                }
            } 
        } 
    }

    return 0;
}

float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b) 
{ 
    return (a.score > b.score); 
}

//NMS处理
int Engine::nmsHandle(std::vector<TargetBox> &tmpBoxes, 
                      std::vector<TargetBox> &dstBoxes,
                      float nmsThresh)
{
    std::vector<int> picked;
    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++) {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            //交集
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            //并集
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;
            if(IoU > nmsThresh) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
    
    for (int i = 0; i < picked.size(); i++) {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

int processInnerBox(std::vector<TargetBox> &dstBoxes) {
    if (dstBoxes.size() >= 3) {
        auto iter = dstBoxes.begin();
        while (iter != (dstBoxes.end()-1)) {
            if (iter->x1 < (iter-1)->x2 && iter->x2 > (iter+1)->x1) {
                iter = dstBoxes.erase(iter);
            }
            else {
                ++iter;
            }
        }
    }
    return 0;
}

bool locationSort(TargetBox &a, TargetBox &b) 
{ 
    return (a.x1 < b.x1); 
}

std::vector<Note> convertBoxs2Notes(std::vector<std::vector<TargetBox>> &boxs, int hop, 
                                    float scaleH, float scaleW, int width) {

    std::vector<Note> notes;
    NoteSate lastState = NoteSate::CLOSE;
    float lastPos = 0;
    for (int i = 0; i < boxs.size(); ++i){
        int offsetPixel = hop * i;
        sort(boxs[i].begin(), boxs[i].end(), locationSort);

        bool first = true;
        float x1, y1, x2, y2;
        for (int j = 0; j < boxs[i].size(); ++j) {
            x1 = boxs[i][j].x1;
            y1 = boxs[i][j].y1;
            x2 = boxs[i][j].x2;
            y2 = boxs[i][j].y2;
            float conf = boxs[i][j].score;
            if (x1 < lastPos - 12.f && x2 < lastPos + 7.f){
                continue;
            }
            float onset = (x1 + offsetPixel) * scaleW;
            float offset = (x2 + offsetPixel) * scaleW;
            float pitch = y1 * scaleH + 21.f;

            if (first && lastState == NoteSate::OPEN && x1 <= lastPos + 7.f) {
                auto tmp = notes.rbegin();
                tmp->offset = offset;
                tmp->pitch = (pitch + tmp->pitch) / 2.f;
                tmp->score = (conf + tmp->score) / 2.f;
                first = false;
            }
            else {
                notes.push_back({onset, offset, pitch, conf});
            }

            if (j == boxs[i].size() - 1) {
                if (x2 < width - 4) {
                    lastState = NoteSate::CLOSE;
                    lastPos = x2 - hop;
                }
                else {
                    lastState = NoteSate::OPEN;
                    lastPos = x1 - hop;
                }
            }
        }
        
    }

    return notes;
}


std::vector<Note> Engine::inferPiece(const std::vector<int16_t>& inAudio, int win, int fft_hop) {
    // infercen the complete piece
    // padding audio n_fft
    int expectLen = inAudio.size() + n_fft / 2;
    expectLen = std::ceil(static_cast<float>(expectLen - win) / fft_hop) * fft_hop + win;
    std::vector<int16_t> audio(expectLen);
    std::copy(inAudio.begin(), inAudio.end(), audio.begin() + n_fft / 2);
    
    int n_segments = (audio.size() - win) / fft_hop + 1;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    std::vector<std::vector<TargetBox>> dstBoxes;
    for(int i = 0; i < n_segments; ++i) {
        std::vector<int16_t> cutAudio(audio.begin() + i * fft_hop, audio.begin() + i * fft_hop + win);

        if (cutAudio.size() != getInputLength()) {
            std::cout << "The audio is not the right size of the model!" << std::endl;
            exit(1);
        }

        featureVectors.clear();
        bool succ = runInference(cutAudio, featureVectors);

        // std::ofstream out;
        // out.open("vector1.bin", std::ios::out | std::ios::binary);
        // for (int i = 0; i < featureVectors[0][0].size(); ++ i){
        //     out.write(reinterpret_cast<const char*>(&featureVectors[0][0][i]), sizeof(float));
        // }
        // out.close();
        
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
        std::vector<TargetBox> tmpBoxes;
        predHandle(featureVectors[0], tmpBoxes, 0., 0., confThresh);
        std::vector<TargetBox> boxes;
        nmsHandle(tmpBoxes, boxes, nmsThresh);
        // std::cout << "nms" << " " << boxes.size() << " " << nmsThresh << std::endl;
        sort(boxes.begin(), boxes.end(), [](TargetBox &a, TargetBox &b) { return a.x1 < b.x1; });
        processInnerBox(boxes);
        dstBoxes.emplace_back(std::move(boxes));
    }

    std::vector<Note> notes = convertBoxs2Notes(dstBoxes, hop, scaleH, scaleW, inputWidth);
    sort(notes.begin(), notes.end(), [](Note &a, Note &b) { return a.onset < b.onset; });
    // for (int id = 0; id < notes.size(); ++id){
    //     std::cout << notes[id].onset << " " << notes[id].offset <<" " << notes[id].pitch << " " << notes[id].score << std::endl;
    // }
    // std::cout << "notes size: " << notes.size() << std::endl;

    return notes;
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
