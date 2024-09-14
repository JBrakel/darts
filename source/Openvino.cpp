#include "Openvino.h"

# include <iostream>
# include <filesystem>
# include "openvino/openvino.hpp"
# include <opencv2/opencv.hpp>

ov::Core core;

Openvino::Openvino(const std::string& path_xml, const std::string& path_bin) {

    // Load the model
    std::shared_ptr<ov::Model> model = core.read_model(path_xml, path_bin);
    ov::CompiledModel compiled_model = core.compile_model(model, m_device_name);
    this->m_infer_request = compiled_model.create_infer_request();

    // Prepare the input tensor
    ov::Tensor input_tensor = m_infer_request.get_input_tensor();
    const ov::Shape& input_shape = input_tensor.get_shape();
    size_t channels = input_shape[1];
    size_t height = input_shape[2];
    size_t width = input_shape[3];
    this->m_target_size = cv::Size(width, height);

}

std::map<int, std::vector<cv::Point>> Openvino::calculateClassCenters(const std::map<int, std::vector<int>>& class_indices, const std::vector<cv::Rect>& boxes) {
    std::map<int, std::vector<cv::Point>> class_centers;

    // Calculate centers for all classes
    for (const auto& class_entry : class_indices) {
        int class_id = class_entry.first;
        const std::vector<int>& indices = class_entry.second;

        std::vector<cv::Point> centers;
        for (int idx : indices) {
            // if (idx >= 0 && idx < boxes.size()) {
                const cv::Rect& box = boxes[idx];
                cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
                centers.push_back(center);
            // }
        }
        class_centers[class_id] = centers;
    }

    return class_centers;
}

void Openvino::drawInitBoundBoxes(std::map<int, std::vector<int>> class_indices, cv::Mat image) {
    for (const auto& class_entry : class_indices) {
        int class_id = class_entry.first;
        const std::vector<int>& indices = class_entry.second;

        cv::Scalar color = m_m_classColors[class_id];

        for (int idx : indices) {
            if (idx >= 0 && idx < m_boxes.size()) {
                const cv::Rect& box = m_boxes[idx];
                cv::rectangle(image, box, color, 2);
            }
        }
    }
}

void Openvino::updateColorsAndTotalCount(int count, cv::Mat& image) {
    std::vector<bool> box_updated(m_boxes.size(), false);  // Track if a box has been updated

    for (const auto& class_entry : m_class_indices) {
        int class_id = class_entry.first;

        if (class_id == 1) {
            for (int idx1 : class_entry.second) {
                if (idx1 >= 0 && idx1 < m_boxes.size()) {
                    const cv::Rect& box1 = m_boxes[idx1];
                    cv::Point center1(box1.x + box1.width / 2, box1.y + box1.height / 2);

                    int closest_box_idx = -1;
                    double min_distance = std::numeric_limits<double>::max();

                    for (const auto& box0_idx : m_class_indices[0]) {
                        if (box0_idx >= 0 && box0_idx < m_boxes.size()) {
                            const cv::Rect& box0 = m_boxes[box0_idx];

                            if (box0.contains(center1)) {
                                // Calculate distance from center1 to the center of box0
                                cv::Point center0(box0.x + box0.width / 2, box0.y + box0.height / 2);
                                double distance = std::sqrt(std::pow(center1.x - center0.x, 2) + std::pow(center1.y - center0.y, 2));

                                if (distance < min_distance) {
                                    min_distance = distance;
                                    closest_box_idx = box0_idx;
                                    hitDetected = true;
                                }
                            }
                        }
                    }

                    if (runGame) {
                        if (closest_box_idx != -1 && !box_updated[closest_box_idx]) {
                            m_m_classColors[0] = cv::Scalar(0, 255, 0);
                            const cv::Rect& closest_box = m_boxes[closest_box_idx];
                            cv::rectangle(image, closest_box, m_m_classColors[0], 2);
                            box_updated[closest_box_idx] = true;
                            count++;
                            // total_count = count;
                            // if(count>total_count){ total_count++;}
                        }
                    }
                }
            }
        }
    }
    total_count = count;
}

void Openvino::displayDetections(std::map<int, std::vector<int>> class_indices, std::vector<cv::Rect> boxes, std::map<int, std::vector<float>> class_scores, cv::Mat& image) {

    // refresh params
    int count = 0;
    hitDetected = false;

    // find centers of bounding boxes
    std::map<int, std::vector<cv::Point>> class_centers = calculateClassCenters(class_indices, boxes);

    // Set up colors for each class
    m_m_classColors[0] = cv::Scalar(0, 0, 255);
    m_m_classColors[1] = cv::Scalar(255, 0, 255);

    // draw bounding boxes in predefined colors
    drawInitBoundBoxes(class_indices,image);

    // update detections
    updateColorsAndTotalCount(count, image);

}


// Formats the image to a square shape
// Copied from YOLO github
cv::Mat Openvino::format_to_square(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

// Get detections function boogaloo
// std::vector<int> CupsOpenvino::get_detections(cv::Mat& image, const cv::Size& target_size, ov::InferRequest& infer_request) {
std::vector<int> Openvino::get_detections(cv::Mat& image) {
    // ################### 1. Preprocessing  ###################

    cv::Mat model_input = format_to_square(image); // Format image to square
    cv::Mat blob;
    cv::dnn::blobFromImage(model_input, blob, 1.0 / 255.0, m_target_size, cv::Scalar(), true, false); // More resizing

    const size_t size_to_copy = blob.total() * blob.elemSize(); // Expected size

    // Get size change
    float x_factor = static_cast<float>(model_input.cols) / m_target_size.width;
    float y_factor = static_cast<float>(model_input.rows) / m_target_size.height;


    // ################### 2. Inference      ###################
    // Input image
    ov::Tensor input_tensor = m_infer_request.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), blob.data, size_to_copy);   // Input the image for model

    // Perform inference
    m_infer_request.infer();

    // Output prediction
    ov::Tensor output_tensor = m_infer_request.get_output_tensor();


    // ################### 3. Postprocessing ###################

    std::vector<int> class_ids;
    // std::vector<cv::Rect> boxes;
    // std::map<int, std::vector<float>> class_scores;
    m_boxes.clear();
    m_class_scores.clear();
    m_class_indices.clear();

    float* data = output_tensor.data<float>(); // #### How does this work
    const ov::Shape& shape = output_tensor.get_shape();
    size_t batch_size = shape[0];
    size_t num_boxes = shape[2];
    size_t attributes = shape[1]; // 4 + num_classes // ####################### TODO : Make classes dynamic

    for (size_t i = 1; i < batch_size + 1; ++i) {
        for (size_t j = 0; j < num_boxes; ++j) {
            float x_center = data[i * num_boxes * 0 + j];
            float y_center = data[i * num_boxes * 1 + j];
            float w = data[i * num_boxes * 2 + j];
            float h = data[i * num_boxes * 3 + j];

            int x1 = int((x_center - w * 0.5) * x_factor);
            int y1 = int((y_center - h * 0.5) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            m_boxes.emplace_back(x1, y1, width, height);
            for (size_t k = 4; k < attributes; ++k) {
                float score = data[i * num_boxes * k + j];
                m_class_scores[k - 4].push_back(score);
            }
        }
    }

    // Apply non-maximum suppression
    // std::map<int, std::vector<int>> class_indices;
    for (size_t i = 4; i < attributes; i++) {
        cv::dnn::NMSBoxes(m_boxes, m_class_scores[i - 4], m_model_score_threshold, m_model_NMS_threshold, m_class_indices[i - 4]);
    }

    std::vector<int> detection_count;
    for (int i = 0; i < m_class_indices.size(); ++i) {
        detection_count.push_back(m_class_indices[i].size());
    }
    frameCount++;

    return detection_count;
}

void Openvino::refresh_count() {
    total_count = 0;
}

void Openvino::set_runGame(bool b) {
    runGame = b;
}

bool Openvino::get_runGame() {
    return runGame;
}

void Openvino::set_total_count(int c) {
    total_count = c;
}

void Openvino::clearBoxesBuffer() {
    updated_boxes.clear();
}

std::vector<cv::Point2f> Openvino::getPointsBoundingBoxes() {
    std::vector<cv::Point2f> points;
    for(auto& bb : m_boxes) {
        int x = bb.x;
        int y = bb.y;
        int w = bb.width;
        int h = bb.height;
        int cx = x + w/2;
        int cy = y + h/2;
        points.emplace_back(cx,cy);
    }
    return points;
}




