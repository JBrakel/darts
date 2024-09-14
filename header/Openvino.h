#ifndef CUPSOPENVINO_H
#define CUPSOPENVINO_H

#include <map>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <set>

class Openvino {
private:
    const float m_model_confidence_threshold = 0.25f;
    const float m_model_score_threshold = 0.45f;
    const float m_model_NMS_threshold = 0.70f;
    const bool m_enable_visualization = true;

    std::vector<cv::Rect> m_boxes;
    std::map<int, std::vector<float>> m_class_scores;
    std::map<int, std::vector<int>> m_class_indices;

    ov::InferRequest m_infer_request;
    cv::Size m_target_size;
    const std::string m_device_name = "CPU";

    std::map<int, cv::Scalar> m_m_classColors;

public:
    int frameCount = 0;
    int total_count = 0;
    int high_score = 0;
    bool runGame = false;
    bool hitDetected = false;
    std::set<int> updated_boxes;

    Openvino(const std::string& path_xml, const std::string& path_bin);
    std::map<int, std::vector<cv::Point>> calculateClassCenters(const std::map<int, std::vector<int>>& class_indices, const std::vector<cv::Rect>& boxes);
    void drawInitBoundBoxes(std::map<int, std::vector<int>> class_indices, cv::Mat imag);
    void updateColorsAndTotalCount(int count, cv::Mat& image);
    void displayDetections(std::map<int, std::vector<int>> class_indices, std::vector<cv::Rect> boxes, std::map<int, std::vector<float>> class_scores, cv::Mat& image);
    cv::Mat format_to_square(const cv::Mat& source);
    std::vector<int> get_detections(cv::Mat& image) ;
    const std::vector<cv::Rect>& get_boxes() const { return m_boxes; }
    std::vector<cv::Point2f> getPointsBoundingBoxes();
    const std::map<int, std::vector<float>>& get_class_scores() const { return m_class_scores; }
    const std::map<int, std::vector<int>>& get_class_indices() const { return m_class_indices; }

    bool getHitDetected() const { return hitDetected; }
    int get_count() const { return total_count; }
    void set_total_count(int c);
    void set_runGame(bool b);
    bool get_runGame();
    void refresh_count();
    void clearBoxesBuffer();
};




#endif //CUPSOPENVINO_H
