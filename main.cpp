#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include <vector>

#include "Board.h"
#include "Config.h"
#include "Output.h"
#include <Openvino.h>
#include "cvui.h"

cv::Point2i mousePos(-1, -1);

cv::VideoCapture start_frame(const std::string& video_path, const int start_frame);
cv::Mat display_frame_nr(cv::Mat& img, const int frame_number);
void mouseCallback(int event, int x, int y, int, void*);
std::vector<cv::Point2f> collectedPoints;

int main() {

    // config
    Config cfg;
    const std::string path_config_json = "../config.json";
    nlohmann::json configData = cfg.importDataJson(path_config_json);

    // project
    const std::string projectName = configData["project"]["projectName"];
    int runNr = configData["project"]["run"];
    const std::string run = "run_" + std::to_string(runNr);

    // paths
    const std::string pathDatasets = configData["paths"]["datasets"];
    const std::string pathProject = pathDatasets + '/' + projectName;
    const std::string pathRun = pathProject + '/' + run;
    const std::string pathVideo = pathRun + "/videos/" + "video_14.mp4";
    const std::string path_xml = configData["paths"]["weights"]["xml"];
    const std::string path_bin = configData["paths"]["weights"]["bin"];

    // dataset
    const std::vector<std::string> classNames = configData["dataset"]["classNames"];

    // output
    Output out;
    std::string outputWindowName = "output";
    cv::namedWindow(outputWindowName, cv::WINDOW_NORMAL);

    // read video
    const int cap_id = configData["output"]["cap_id"];
    // cv::VideoCapture cap(cap_id);
    cv::VideoCapture cap = start_frame(pathVideo,0);
    cv::Mat frame;

    if (!cap.isOpened()) {
        return -1;
    }

    // init board
    Board board;

    // helper function
    cv::setMouseCallback(outputWindowName, mouseCallback);

    // init Openvino
    Openvino vino(path_xml, path_bin);
    std::vector<cv::Rect> boxes;
    std::map<int, std::vector<float>> class_scores;
    std::map<int, std::vector<int>> class_indices;

    // mapping
    std::map<int, std::string> classMap = {
        {0, "db"},
        {1, "d20_p0"},
        {2, "d20_p1"},
        {3, "d6_p0"},
        {4, "d6_p1"},
        {5, "d3_p0"},
        {6, "d3_p1"},
        {7, "d11_p0"},
        {8, "d11_p1"}
    };

    while(true) {
        cap >> frame;

        const double sizeFrameWarped = frame.rows;
        int frame_number = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
        display_frame_nr(frame,frame_number);

        if (frame.empty()) {
            break;
        }

        // calc all points on board
        board.calcPointsFullBoard();

        // get source points
        // detect cups and sports balls using openvino
        vino.get_detections(frame);
        class_indices = vino.get_class_indices();
        boxes = vino.get_boxes();
        std::map<int, std::vector<cv::Point>> pointsBoundingBoxes = vino.calculateClassCenters(class_indices, boxes);

        std::vector<cv::Point2f> srcPoints;
        std::vector<cv::Point2f> dstPoints;

        int d20_count = 0;
        int d6_count = 0;
        int d3_count = 0;
        int d11_count = 0;
        int db_count = 0;
        int total_count;

        for (const auto& classPair : classMap) {
            int classID = classPair.first;

            // Check if the class ID exists in pointsBoundingBoxes
            auto it = pointsBoundingBoxes.find(classID);
            if (it != pointsBoundingBoxes.end()) {
                const std::vector<cv::Point>& points = it->second;

                for (const auto& point : points) {
                    // if (classMap[classID] == "db" && db_count == 0) {
                    //     srcPoints.emplace_back(point.x, point.y);
                    //     dstPoints.push_back(board.getPositionBullseye());
                    //     db_count++;
                    // }
                    if ((classMap[classID] == "d20_p0" || classMap[classID] == "d20_p1") && d20_count == 0) {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 0);
                        srcPoints.emplace_back(point.x, point.y);
                        dstPoints.push_back(classMap[classID] == "d20_p0" ? temp[0] : temp[1]);
                        d20_count++;
                    }
                    else if ((classMap[classID] == "d6_p0" || classMap[classID] == "d6_p1") && d6_count == 0) {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 6);
                        srcPoints.emplace_back(point.x, point.y);
                        dstPoints.push_back(classMap[classID] == "d6_p0" ? temp[0] : temp[1]);
                        d6_count++;
                    }
                    else if ((classMap[classID] == "d3_p0" || classMap[classID] == "d3_p1") && d3_count == 0) {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 3);
                        srcPoints.emplace_back(point.x, point.y);
                        dstPoints.push_back(classMap[classID] == "d3_p0" ? temp[0] : temp[1]);
                        d3_count++;
                    }
                    else if ((classMap[classID] == "d11_p0" || classMap[classID] == "d11_p1") && d11_count == 0) {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 11);
                        srcPoints.emplace_back(point.x, point.y);
                        dstPoints.push_back(classMap[classID] == "d11_p0" ? temp[0] : temp[1]);
                        d11_count++;
                    }
                }
            }
        }

        total_count = d20_count + d6_count + d3_count + d11_count + db_count;

        cv::Mat frameWarped(frame.size(), frame.type(), cv::Scalar(0));
        if (total_count >= 4) {

            // homography
            cv::Mat frameResized(cv::Size(sizeFrameWarped,sizeFrameWarped), frame.type());
            cv::Mat homography = cv::findHomography(srcPoints, dstPoints, cv::RANSAC);
            cv::warpPerspective(frame, frameWarped, homography, cv::Size(sizeFrameWarped,sizeFrameWarped));

            // create masks
            cv::Mat maskBinary(frame.size(), CV_8U, cv::Scalar(255));
            cv::Mat maskBinaryFieldsHit(frame.size(), CV_8U, cv::Scalar(255));

            // calc board
            board.drawBoard(frameWarped);
            board.drawBoard(maskBinary);

            // draw fields
            // board.drawField(frameWarped,"t20");
            // board.drawField(maskBinaryFiels,"sb");
            // board.drawField(maskBinaryFiels,"d15");

            // inverse homography
            cv::Mat homographyInverse = homography.inv();

            // invert mask
            cv::Mat maskBinaryInverted;
            cv::bitwise_not(maskBinary, maskBinaryInverted);
            cv::warpPerspective(maskBinaryInverted, maskBinaryInverted, homographyInverse, cv::Size(frame.cols,frame.rows));

            // apply closing -> smoother edges
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)); // Change size as needed
            cv::morphologyEx(maskBinaryInverted, maskBinaryInverted, cv::MORPH_CLOSE, element);

            // draw colored board considering perspective
            for (int y = 0; y < maskBinaryInverted.rows; ++y) {
                for (int x = 0; x < maskBinaryInverted.cols; ++x) {
                    if (maskBinaryInverted.at<uchar>(y, x) != 0) {
                        frame.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                    }
                }
            }

            // repeat and draw fields that are hit
            cv::Mat maskBinaryInvertedFieldHit;
            cv::bitwise_not(maskBinaryFieldsHit, maskBinaryInvertedFieldHit);
            cv::warpPerspective(maskBinaryInvertedFieldHit, maskBinaryInvertedFieldHit, homographyInverse, cv::Size(frame.cols,frame.rows));
            cv::morphologyEx(maskBinaryInvertedFieldHit, maskBinaryInvertedFieldHit, cv::MORPH_CLOSE, element);

            for (int y = 0; y < maskBinaryInvertedFieldHit.rows; ++y) {
                for (int x = 0; x < maskBinaryInvertedFieldHit.cols; ++x) {
                    if (maskBinaryInvertedFieldHit.at<uchar>(y, x) != 0) {
                        frame.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 255);
                    }
                }
            }


            for (size_t i = 0; i < dstPoints.size(); ++i) {
                cv::Scalar color = (0, 0, 255);
                cv::circle(frameWarped, dstPoints[i], 9, color, -1);
            }
        }

        // draw bounding boxes
        for (const auto& pair : pointsBoundingBoxes) {
            int classID = pair.first;
            const std::vector<cv::Point>& points = pair.second;
            cv::Scalar color = cv::Scalar(255, 0, 0);

            for (const auto& point : points) {
                if (std::find(srcPoints.begin(), srcPoints.end(), cv::Point2f(point.x, point.y)) != srcPoints.end()) {
                    int boxWidth = 25;
                    int boxHeight = 25;
                    cv::Point topLeft(point.x - boxWidth / 2, point.y - boxHeight / 2);
                    cv::Point bottomRight(point.x + boxWidth / 2, point.y + boxHeight / 2);
                    cv::rectangle(frame, topLeft, bottomRight, color, 3);

                    cv::Point textPosition(topLeft.x, topLeft.y - 5);
                    // cv::putText(frame, classMap[classID], textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, color, 3);

                }
            }
        }


        // create gui
        // create output window
        cv::Mat outputWindow(frame.rows, frame.cols+sizeFrameWarped+1, frame.type());
        outputWindow.setTo(cv::Scalar(75, 75, 75));

        // add frame to outputwindow
        frame.copyTo(outputWindow(cv::Rect(0, 0, frame.cols, frame.rows)));

        // add warped frame to outputwindow
        cv::resize(frameWarped, frameWarped, cv::Size(sizeFrameWarped, sizeFrameWarped));
        frameWarped.copyTo(outputWindow(cv::Rect(frame.cols+1, 0, frameWarped.cols, frameWarped.rows)));

        // output
        cv::imshow(outputWindowName, outputWindow);

        int key = cv::waitKey(0);
        if (key == 'q') {
            break;
        }

    }
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Collected Points:\n";
    for (const auto& point : collectedPoints) {
        std::cout << point << std::endl;
    }

    return 0;
}

// more functions
void mouseCallback(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Record the point on left mouse button click
        collectedPoints.emplace_back(x, y);
        std::cout << "Point collected: (" << x << ", " << y << ")\n";
    }
}

cv::VideoCapture start_frame(const std::string& video_path, const int start_frame) {
    cv::VideoCapture cap(video_path);
    if(!cap.isOpened()) {
        std::cout << "Error: Could not open video." << std::endl;
        return {};
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
    return cap;
}

cv::Mat display_frame_nr(cv::Mat& img, const int frame_number) {
    char text[20];
    snprintf(text, sizeof(text), "Frame: %d", frame_number);
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Point bottom_left_corner(10, img.rows - 10);
    const double font_scale = 1.5;
    const cv::Scalar font_color(255,255,255);
    const int line_type = 2;
    cv::putText(img,
                text,
                bottom_left_corner,
                font,
                font_scale,
                font_color,
                line_type);
    return img;
}
