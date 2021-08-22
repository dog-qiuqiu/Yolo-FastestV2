#include "yolo-fastestv2.h"

int main()
{   yoloFastestv2 api;

    api.loadModel("./model/yolo-fastestv2.param",
                  "./model/yolo-fastestv2.bin");

    cv::Mat cvImg = cv::imread("test.jpg");

    std::vector<TargetBox> boxes;
    api.detection(cvImg, boxes);

    for (int i = 0; i < boxes.size(); i++) {
        std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<" "<<boxes[i].y2
                 <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;
        
        cv::rectangle (cvImg, cv::Point(boxes[i].x1, boxes[i].y1), 
                       cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
    }
    
    cv::imwrite("output.png", cvImg);

    return 0;
}