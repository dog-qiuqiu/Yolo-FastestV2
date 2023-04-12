#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>
#include "audio.h"
#include "musicyolo.h"

typedef std::chrono::high_resolution_clock Clock;
namespace fs = std::experimental::filesystem;

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

int writeNotes(const std::string &filePath, const std::vector<Note> &notes){
    std::ofstream out;
    out.open(filePath, std::ios::out);
    for(const Note &note: notes){
        std::string noteString = string_format("%-6.3f\t%-6.3f\t%-7.3f\t%.3f\n", note.onset, 
                                               note.offset, note.pitch, note.score);
        out.write(reinterpret_cast<const char*>(noteString.data()), sizeof(char)*noteString.size());
    }
    out.close();
}

int main(int argc, char *argv[]) {
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

    const std::string audioDir(argv[1]);
    const std::string saveDir(argv[2]);

    if(!fs::exists(saveDir)){
        if(!fs::create_directory(saveDir)){
            std::cout << "create directory failed!" << std::endl;
        }
    }

    std::vector<int16_t> cpuAudio;
    std::vector<Note> notes;
    for(const auto &entry: fs::directory_iterator(audioDir)){
        const std::string &inPath = entry.path();
        std::cout << inPath << std::endl;
        cpuAudio.clear();
        notes.clear();
        cpuAudio = audioRead(inPath); // [0, 1]
        notes = engine.inferPiece(cpuAudio, 80448, 24000);
        std::string base_filename = inPath.substr(inPath.find_last_of("/\\") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string file_without_extension = base_filename.substr(0, p);
        std::string savePath = fs::path(saveDir) / fs::path(file_without_extension + ".txt");
        writeNotes(savePath, notes);
        std::cout << savePath << std::endl;
    } 

    return 0;
}