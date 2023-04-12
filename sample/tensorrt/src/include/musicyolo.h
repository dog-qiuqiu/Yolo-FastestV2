#ifndef TENSORRT_H_
#define TENSORRT_H_

#include "NvInfer.h"
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>

// 参考代码 https://github.com/cyrusbehr/tensorrt-cpp-api/blob/main/src/engine.h

class TargetBox
{
private:
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    float area() { return getWidth() * getHeight(); };
};

class Note
{
public:
    Note(float on, float off, float p, float s): onset(on), offset(off), pitch(p), score(s) {}
    float onset;
    float offset;
    float pitch;
    float score;
};

// Precision used for GPU inference
enum class Precision {
    FP32,
    FP16
};

// Note state for open and close
enum class NoteSate {
    CLOSE,
    OPEN
};

// Options for the network
struct Options {
    bool doesSupportDynamicBatchSize = true;
    // Precision to use for GPU inference. 16 bit is faster but may reduce accuracy.
    Precision precision = Precision::FP32;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 4000000000;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(const Options& options);
    ~Engine();
    // Build the network
    bool build(std::string onnxModelPath);
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference.
    bool runInference(const std::vector<int16_t> &input, std::vector<std::vector<std::vector<float>>>& featureVectors);
    std::vector<Note> inferPiece(const std::vector<int16_t>& audio, int win, int hop);

    int32_t getInputLength() const { return m_inputL; };

private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    bool doesFileExist(const std::string& filepath);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;

    int32_t m_inputL;

    inline void checkCudaErrorCode(cudaError_t code);

    int predHandle(const std::vector<std::vector<float>> &out, std::vector<TargetBox> &dstBoxes, 
                   const float scaleW, const float scaleH, const float thresh);
    int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes, float nmsThresh);

    int inputWidth;
    int inputHeight;
    float scaleH;
    float scaleW;
    int hop;
    int n_fft;
    float confThresh;
    float nmsThresh;
    int numAnchor;
    std::vector<int> outdims;
    std::vector<float> anchor;
};

#endif