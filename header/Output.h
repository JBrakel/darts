#ifndef OUTPUT_H
#define OUTPUT_H

#include <opencv2/opencv.hpp>

class Output {

public:
    cv::Point calculateCenteredCoords(int windowWidth, int windowHeight, bool singleScreen);
};

#endif // OUTPUT_H
