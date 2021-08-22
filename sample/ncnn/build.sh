g++ -g -o demo demo.cpp src/yolo-fastestv2.cpp -I src/include -I include/ncnn lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp
