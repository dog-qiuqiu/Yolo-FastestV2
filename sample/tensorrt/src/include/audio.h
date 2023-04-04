#ifndef AUDIO_H_
#define AUDIO_H_

#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>

struct WavHeader
{
    char riff[4]; // RIFF string
    int32_t fileSize; // total file size minus 8 bytes
    char wave[4]; // WAVE string
    char fmt[4]; // fmt string with trailing null char
    int32_t fmtSize; // length of the format data
    int16_t format; // format type, 1 = PCM, 3 = IEEE float, 6 = 8bit A law, 7 = 8bit mu law, ...
    int16_t channels; // number of channels (mono/stereo)
    int32_t sampleRate; // sampling rate
    int32_t byteRate; // SampleRate * NumChannels * BitsPerSample/8
    int16_t blockAlign; // NumChannels * BitsPerSample/8
    int16_t bitsPerSample; // bits per sample, 8- 8bits, 16 - 16 bits etc
    char data[4]; // DATA string or FLLR string
    int32_t dataSize; // NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
};

std::vector<int16_t> audioRead(const std::string &filepath)
{
    WavHeader header;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("Unable to read wav head");
    }

    assert(header.bitsPerSample == 16);
    std::vector<int16_t> audioData(header.dataSize/2);
    if (!file.read(reinterpret_cast<char*>(audioData.data()), header.dataSize/2 * sizeof(int16_t))) {
        throw std::runtime_error("Unable to read wav data");
    }
    file.close();
    // std::vector<float> audioDataFloat(header.dataSize/2);
    // std::transform(audioData.begin(), audioData.end(), audioDataFloat.begin(), 
    //                [](const int16_t& val) -> float { return static_cast<float>(val) / 32768; })
    return audioData;
}



#endif