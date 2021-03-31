#include <opencv2/opencv.hpp>
#include <string>

class VideoProcessor {

    public:
        std::string input;
        std::string output;
        std::function<void(cv::Mat&, cv::Mat&)> fcn;

    public:
        VideoProcessor(
            std::string in, 
            std::string out, 
            std::function<void(cv::Mat&, cv::Mat&)> fcn
        );

        ~VideoProcessor();
        void execute();
    
    private:
        cv::VideoCapture instream;
        cv::VideoWriter outstream;

};