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
    // const std::string onnxModelpath = "../../model/musicyolo-opt.onnx";
    const std::string onnxModelpath = "/home/data/wxk/Yolo-FastestV2/model/musicyolo.onnx";

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
    const std::string inputAudio = "/home/data/SSVD-v2.0/test16k/100135.wav";
    std::vector<int16_t> cpuAudio = audioRead(inputAudio); // [0, 1]

    std::vector<Note> notes = engine.inferPiece(cpuAudio, 80448, 24000);

    return 0;
}