#include "imageio.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
};

VideoProcessor::VideoProcessor(
    std::string in,
    std::string out,
    std::function<void(cv::Mat&, cv::Mat&)> fcn){
    /*
        VideoProcessor - Utility class to read frames from IN, process using the FCN specified and write to OUT.
    */

    this->input  = in;
    this->output = out;
    this->fcn = fcn;
    
    // Input stream:
    instream  = cv::VideoCapture(in);

    // Define output dimensions:
    cv::Size dim(
        instream.get(cv::CAP_PROP_FRAME_WIDTH),
        instream.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    //Open the stream:
    outstream = cv::VideoWriter(
        out, cv::VideoWriter::fourcc('M','J','P','G'), 30, dim
    );
};

VideoProcessor::~VideoProcessor(){
    // Close the stream if it is still open:
    if (instream.isOpened()){ 
        instream.release(); 
        }
    if (outstream.isOpened()){ 
        outstream.release(); 
        }
};

void VideoProcessor::execute(){
    /**
     * @brief TODO
     * 
     */

    cv::Mat in_frame, out_frame;
    int frame_id (0);
    double progress;

    while(1) {
        instream >> in_frame;
        if (in_frame.empty())
            break;
        
        // Execute the function:
        fcn(in_frame, out_frame);
        
        // Write the output to file:
        outstream.write(out_frame);

        // Print update:
        progress = double(frame_id++)/double(instream.get(cv::CAP_PROP_FRAME_COUNT));
        printProgress(progress);
    }
};