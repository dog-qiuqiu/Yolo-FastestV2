#include <opencv2/opencv.hpp>
#include <chrono>
#include "audio.h"
#include "musicyolo.h"

typedef std::chrono::high_resolution_clock Clock;

int main() {
    // Specify our GPU inference configuration options
    Options options;
    // TODO: If your model only supports a static batch size
    options.doesSupportDynamicBatchSize = false;
    options.precision = Precision::FP32; // Use fp16 precision for faster inference.

    if (options.doesSupportDynamicBatchSize) {
        options.optBatchSize = 4;
        options.maxBatchSize = 16;
    }

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    // If model only specifies a static batch size, must set the above variable doesSupportDynamicBatchSize to false.
    const std::string onnxModelpath = "/home/data/wxk/Yolo-FastestV2/model/musicyolo-opt.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Lets use a batch size which matches that which we set the Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // read audio wav
    const std::string inputAudio = "/home/data/wxk/OMAP2/test/wav/01_01.wav";
    std::vector<int16_t> cpuAudio = audioRead(inputAudio); // [0, 1]
    std::vector<int16_t> cutCpuAudio(cpuAudio.begin(), cpuAudio.begin() + 80448);
    cv::Mat srcAudio(1, cutCpuAudio.size(), CV_16SC1, reinterpret_cast<void*>(cutCpuAudio.data()));

    cv::cuda::GpuMat audio;
    audio.upload(srcAudio); // 80448的数据塞进去变成了 [80448 x 1]的数据，其中rows = channels = 1, cols = 80448 size() = [80448 x 1]

    // TODO: If the model expects a different input size, resize it here.
    // You can choose to resize by scaling, adding padding, or a conbination of the two in order to maintain the aspect ratio
    // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
    // If you are running the sample code using the suggested model, then the input image already has the correct size.
    std::cout << "input length " << engine.getInputLength() << std::endl;;
    if (cutCpuAudio.size() != engine.getInputLength()) {
        std::cout << "The audio is not the right size of the model!" << std::endl;
        return -1;
    }
    
    
    std::vector<cv::cuda::GpuMat> audios;
    audios.push_back(audio);

    // Discard the first inference time as it takes longer
    std::vector<std::vector<std::vector<float>>> featureVectors;
    succ = engine.runInference(audios, featureVectors);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(audios, featureVectors);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(audios.size()) <<
    " ms, for batch size of: " << audios.size() << std::endl;

    // Print the feature vectors
    for (int batch = 0; batch < featureVectors.size(); ++batch) {
        for (int outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }

    // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}