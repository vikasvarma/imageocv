#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <numeric>
#include "findlane.hpp"

double moving_avg(std::vector<double> &values, int N){
    // Find the moving average of the last N items of a vector:
    double avg, total;
    if (values.size() > N) {
        total = std::accumulate(values.end()-30, values.end(), 0.0);
        avg   = total/double(N);
    } else {
        total = std::accumulate(values.begin(), values.end(), 0.0);
        avg   = total/double(values.size());
    }

    return avg;
};

cv::Vec4i calculate_coord(double slope, double intercept, int rows, int cols){
    /**
     * @brief calculate_coord returns the extents of the line segment
     *        to be drawn on the frame.
     */

    int x0,y0,x1,y1;

    y0 = int(0.65*rows);
    y1 = int(rows);
    x0 = int((1.0*y0 - intercept)/slope);
    x1 = int((1.0*y1 - intercept)/slope);

    return cv::Vec4i(x0,y0,x1,y1);
};

LaneDetector::LaneDetector(){
    /**
     * @brief LANEDETECTOR - Lane detection module.
     * 
     */
};

LaneDetector::~LaneDetector(){
    /**
     * @brief LANEDETECTOR - Lane detection module.
     * 
     */
    
    // Nothing to do here.
};

void LaneDetector::detect(cv::Mat &frame, cv::Mat &out){
    /**
     * @brief DETECT - Identify the segmentation for lanes in input FRAME.
     * 
     * Input  - A cv::Mat frame containing a road scene.
     * Output - Reference to a cv::Mat object containing the segmented input
     *          frame.
     * 
     */

    cv::Mat gray, filtered, roi;
    std::vector<cv::Vec4i> lines;

    // Filter the frame to have 
    filter_colors(frame, filtered);
    apply_roi(filtered, roi);

   // Convert to grayscale image:
    cv::cvtColor(roi, gray, cv::COLOR_RGB2GRAY);

    // Reduce noise in edge detection through a gaussial blur
    cv::blur(gray, gray, kernel_size);

    // Do canny edge detection:
    cv::Canny(gray, out, canny_low, canny_high);

    // Now, apply hough transform to find the bounds of the line segment 
    // detected by the transform.
    cv::HoughLinesP(out, lines, rho, theta, thr, 100, 50);

    // Overlay detected lines on the frame and construct output frame:
    draw_lines(frame, out, lines);
};

void LaneDetector::apply_roi(cv::Mat &frame, cv::Mat &out){
    /**
     * @brief apply_roi(FRAME) narrows the region of lane detection by applying 
     * filtering image content outside a region of interest.\
     */

    // Define the bounds of the ROI:
    int rows (frame.rows), cols (frame.cols);
    cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
    
    std::vector<cv::Point2i> roi {
        {        int(0),     int(rows)},
        {     int(cols),     int(rows)},
        {int(0.55*cols), int(0.6*rows)},
        {int(0.45*cols), int(0.6*rows)}
    };
    
    // Fill polygon with white pixels and apply mask:
    const cv::Scalar white (1,frame.channels(),CV_8U, 255);
    cv::fillPoly(mask, roi, white);
    cv::bitwise_and(frame, frame, out, mask = mask);
};

void LaneDetector::filter_colors(cv::Mat &frame, cv::Mat &out){
    /**
     * @brief apply_roi(FRAME) narrows the region of lane detection by applying 
     * filtering image content outside a region of interest.
     * 
     * Road lanes are either yellow or white, so rest of the colors can be 
     * filtered to avoid false positive detection.
     */

    cv::Mat hls, yellow_mask, white_mask, mask;
    cv::Scalar LOW_WHITE   {  0, 190,   0};
    cv::Scalar HIGH_WHITE  {255, 255, 255};
    cv::Scalar LOW_YELLOW  { 20,   0,  90}; 
    cv::Scalar HIGH_YELLOW { 30, 255, 255};

    // Convert frame to HLS colorspace:
    cv::cvtColor(frame, hls, cv::COLOR_BGR2HLS);

    // Filter image to be within specified color ranges and apply a bitwise OR
    // operation to compute the combined image mask.
    cv::inRange(hls, LOW_YELLOW, HIGH_YELLOW, yellow_mask);
    cv::inRange(hls, LOW_WHITE, HIGH_WHITE, white_mask);
    cv::bitwise_or(yellow_mask, white_mask, mask);
    cv::bitwise_and(frame, frame, out, mask = mask);
};

void LaneDetector::draw_lines(
    cv::Mat &frame, cv::Mat &out, std::vector<cv::Vec4i> &lines){
    /**
     * @brief draw_lines will select valid lanes from line segments detected
     *        using hough transform and draw them on the frame to produce
     *        output video feed.
     */

    // Lane labels:
    cv::Mat labels (frame.size(), frame.type(), cv::Scalar(0));

    // y = mx + b
    double m, b;
    for(cv::Vec4i segment : lines){
        m = double(segment[1]-segment[3])/double(segment[0]-segment[2]);
        b = 1.0*segment[3] - m*segment[2];

        if (m > 0.2){
            // Positive slope, right lane:
            right_slope.push_back(m);
            right_intercept.push_back(b);

        } else if (m < -0.2){
            // Negative slope, left lane:
            left_slope.push_back(m);
            left_intercept.push_back(b);
        }
    }

    // Calculate average of the last 30 frames of the video:
    // NOTE: This helps stabilize line estimates across frames.
    double lavg_m, ravg_m, lavg_b, ravg_b;
    lavg_m = moving_avg(left_slope, 30);
    ravg_m = moving_avg(right_slope, 30);
    lavg_b = moving_avg(left_intercept, 30);
    ravg_b = moving_avg(right_intercept, 30);

    // Find the intercepts:
    cv::Vec4i left, right;
    left  = calculate_coord(lavg_m, lavg_b, frame.rows, frame.cols);
    right = calculate_coord(ravg_m, ravg_b, frame.rows, frame.cols);

    std::vector<cv::Point2i> lane {
        { left[0],  left[1]},
        { left[2],  left[3]},
        {right[2], right[3]},
        {right[0], right[1]}
    };

    cv::fillPoly(labels, lane, cv::Scalar(0,0,255));
    cv::line(labels, lane[0], lane[1], left_color, 10);
    cv::line(labels, lane[3], lane[2], right_color, 10);

    //Now, overlay the labels on the image:
    cv::addWeighted(frame, 0.8, labels, 1.0, 0.0, out);
};