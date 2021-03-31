#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "findlane.hpp"
#include "imageio.hpp"

using std::cout;
using std::endl;

int main(int argc, char** argv) {

    // Validate that two input arguments are specified:
    if (argc != 3){
        throw std::invalid_argument("Invalid input arguments specified.");
    }

    // Allocate files:
    std::string in_file (argv[0]);
    std::string out_file (argv[1]);

    std::function<void(cv::Mat&, cv::Mat&)> fcn = 
    [&](cv::Mat &in, cv::Mat &out) { 
        LaneDetector detector;
        detector.detect(in, out); 
    };

    VideoProcessor processor(in_file, out_file, fcn);
    processor.execute();

    return 0;
}