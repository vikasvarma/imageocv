#include <opencv2/opencv.hpp>
#include <string>

class LaneDetector {

    public:
        LaneDetector();
        ~LaneDetector();
        void detect(cv::Mat &frame, cv::Mat &out);

    private:
        double canny_low   {       50};
        double canny_high  {      150};
        double rho         {        2};
        double theta       {CV_PI/180};
           int thr         {      100};
      cv::Size kernel_size {      5,5};

        std::vector<double> left_slope;
        std::vector<double> left_intercept;
        std::vector<double> right_slope;
        std::vector<double> right_intercept;

        cv::Vec3i left_color  {255,0,0};
        cv::Vec3i right_color {0,255,0};

    private:
        void apply_roi(cv::Mat &frame, cv::Mat &out);
        void filter_colors(cv::Mat &frame, cv::Mat &out);
        void draw_lines(cv::Mat &frame, cv::Mat &out, std::vector<cv::Vec4i> &lines);
};