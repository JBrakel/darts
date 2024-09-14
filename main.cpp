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
#include "Projection.h"
#include <Openvino.h>
#include "cvui.h"
#include "../../../../../Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.5.sdk/System/Library/Frameworks/CoreText.framework/Headers/CTFrame.h"

// todo datensatz labeln
// todo openvino importieren


cv::Point2i mousePos(-1, -1);

cv::VideoCapture start_frame(const std::string& video_path, const int start_frame);
cv::Mat display_frame_nr(cv::Mat& img, const int frame_number);
// void mouseCallback(int event, int x, int y, int, void*);
std::vector<cv::Point2f> collectedPoints;

// Mouse callback function
void mouseCallback(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Record the point on left mouse button click
        collectedPoints.emplace_back(x, y);
        std::cout << "Point collected: (" << x << ", " << y << ")\n";
    }
}


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
    const int cap_id = 0;
    cv::VideoCapture cap(cap_id);
    // cv::VideoCapture cap(pathVideo);
    // cv::VideoCapture cap = start_frame(pathVideo,0);
    cv::Mat frame;

    if (!cap.isOpened()) {
        return -1;
    }

    // init board
    Board board;

    // init projection
    Projection proj;

    // helper function
    cv::setMouseCallback(outputWindowName, mouseCallback);
    // cv::setMouseCallback("warped", mouseCallback);

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

    cv::Mat prevGray;
    std::vector<cv::Point2f> prevPoints;  // Previous points detected by YOLO or Optical Flow
    std::vector<cv::Point2f> nextPoints;  // Next points after optical flow calculation
    std::vector<uchar> status;            // Status vector to indicate if flow is found
    std::vector<float> err;               // Error vector
    std::map<int, std::vector<cv::Point>> opticalFlowPoints;

    while(true) {
        cap >> frame;


        const double sizeFrameWarped = frame.rows;

        int frame_number = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
        display_frame_nr(frame,frame_number);

        if (frame.empty()) {
            break;
        }

        // Convert to grayscale for optical flow
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);


        // First frame: Initialize the previous frame and YOLO detections
        // if (prevGray.empty()) {
        //     gray.copyTo(prevGray);
        //
        //     // Detect points using YOLO in the initial frame
        //     vino.get_detections(frame);
        //     class_indices = vino.get_class_indices();
        //     boxes = vino.get_boxes();
        //     std::map<int, std::vector<cv::Point>> pointsBoundingBoxes = vino.calculateClassCenters(class_indices, boxes);
        //
        //     // Convert points to prevPoints for initial tracking
        //     for (const auto& classPair : classMap) {
        //         int classID = classPair.first;
        //         auto it = pointsBoundingBoxes.find(classID);
        //         if (it != pointsBoundingBoxes.end()) {
        //             const std::vector<cv::Point>& points = it->second;
        //             for (const auto& point : points) {
        //                 prevPoints.emplace_back(point.x, point.y);  // Initialize points for tracking
        //             }
        //         }
        //     }
        //
        //     continue;  // Skip to the next frame
        // }
        //
        // // Calculate Optical Flow if previous points are available
        // if (!prevPoints.empty()) {
        //     cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);
        //
        //     // Clear srcPoints to repopulate it
        //     std::vector<cv::Point2f> srcPoints;
        //
        //     // Iterate through the class indices map
        //     for (const auto& entry : class_indices) {
        //         int classID = entry.first;  // Class ID
        //         const std::vector<int>& indices = entry.second;  // Corresponding indices
        //
        //         // Process each index for the current class ID
        //         for (size_t i = 0; i < indices.size(); i++) {
        //             if (status[i] == 1) {  // Point tracked successfully
        //                 srcPoints.push_back(nextPoints[i]);
        //
        //                 // Update opticalFlowPoints map with tracked points
        //                 opticalFlowPoints[classID].emplace_back(cv::Point(nextPoints[i].x, nextPoints[i].y));
        //             }
        //         }
        //     }
        //
        //     // Update previous points and frames
        //     prevPoints = srcPoints;  // Replace old points with new tracked points
        //     gray.copyTo(prevGray);   // Update previous frame
        // }


        // calc all points on board
        board.calcPointsFullBoard();

        ////////////////////////////////////////////// import srcpoint from openvino
        // get source points
        // detect cups and sports balls using openvino
        vino.get_detections(frame);
        class_indices = vino.get_class_indices();

        boxes = vino.get_boxes();
        std::map<int, std::vector<cv::Point>> pointsBoundingBoxes = vino.calculateClassCenters(class_indices, boxes);

        // for (const auto& classPair : classMap) {
        //     int classID = classPair.first;
        //
        //     // Check if the class ID is missing in pointsBoundingBoxes
        //     if (pointsBoundingBoxes.find(classID) == pointsBoundingBoxes.end() && !opticalFlowPoints[classID].empty()) {
        //         // Use Optical Flow points as fallback
        //         pointsBoundingBoxes[classID] = opticalFlowPoints[classID];
        //     }
        // }

        std::vector<cv::Point2f> srcPoints;
        std::vector<cv::Point2f> dstPoints;
        std::array<int, 4> dstFields= {0, 6, 3, 11};

        int d20_count = 0;
        int d6_count = 0;
        int d3_count = 0;
        int d11_count = 0;
        int db_count = 0;
        int total_count;

        // Iterate over classMap to ensure order
        for (const auto& classPair : classMap) {
            int classID = classPair.first;

            // Check if the class ID exists in pointsBoundingBoxes
            auto it = pointsBoundingBoxes.find(classID);
            if (it != pointsBoundingBoxes.end()) {
                const std::vector<cv::Point>& points = it->second;

                // Add each point to srcPoints in cv::Point2f format
                for (const auto& point : points) {
                    srcPoints.emplace_back(point.x, point.y);

                    // Now add the corresponding point to dstPoints
                    if (classMap[classID] == "db") {
                        dstPoints.push_back(board.getPositionBullseye());
                        db_count = 1;
                    }
                    else if (classMap[classID] == "d20_p0") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 0);
                        dstPoints.push_back(temp[0]);
                        d20_count = 1;
                    }
                    else if (classMap[classID] == "d20_p1") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 0);
                        dstPoints.push_back(temp[1]);
                        d20_count = 1;
                    }
                    else if (classMap[classID] == "d6_p0") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 6);
                        dstPoints.push_back(temp[0]);
                        d6_count = 1;
                    }
                    else if (classMap[classID] == "d6_p1") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 6);
                        dstPoints.push_back(temp[1]);
                        d6_count = 1;
                    }
                    else if (classMap[classID] == "d3_p0") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 3);
                        dstPoints.push_back(temp[0]);
                        d3_count = 1;
                    }
                    else if (classMap[classID] == "d3_p1") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 3);
                        dstPoints.push_back(temp[1]);
                        d3_count = 1;
                    }
                    else if (classMap[classID] == "d11_p0") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 11);
                        dstPoints.push_back(temp[0]);
                        d11_count = 1;
                    }
                    else if (classMap[classID] == "d11_p1") {
                        std::array<cv::Point, 4> temp = board.getPointsFromField("d", 11);
                        dstPoints.push_back(temp[1]);
                        d11_count = 1;
                    }
                }
            }
        }

        total_count = d20_count + d6_count + d3_count + d11_count + db_count;

        //////////////////////////////////////////////
        cv::Mat frameWarped(frame.size(), frame.type(), cv::Scalar(0));
        // if(srcPoints.size()>3) {
        if (total_count >=4) {


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
            // board.drawField(maskBinary,"t20");
            // board.drawField(maskBinaryFiels,"sb");
            // board.drawField(maskBinaryFiels,"d15");


            // inverse homography
            // cv::Mat homographyInverse = cv::findHomography(dstPoints, srcPoints, cv::RANSAC);
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
                    // if (maskInv.at<uchar>(y, x) == 255) {
                    if (maskBinaryInverted.at<uchar>(y, x) != 0) {
                        frame.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Set pixel to red (BGR format)
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
                        frame.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 255); // Set pixel to red (BGR format)
                    }
                }
            }


            std::vector<cv::Scalar> colors = {
                cv::Scalar(0, 0, 255),   // Red
                cv::Scalar(255, 0, 0),   // Blue
                cv::Scalar(0, 155, 0),   // Green
                cv::Scalar(0, 255, 255), // Yellow
                cv::Scalar(255, 0, 255), // Magenta
                cv::Scalar(255, 255, 0), // Cyan
                cv::Scalar(128, 128, 128) // Gray (add more colors if needed)
            };

            for (size_t i = 0; i < dstPoints.size(); ++i) {
                cv::Scalar color = colors[i % colors.size()];
                cv::circle(frame, srcPoints[i], 7, color, -1); // Draw filled circle
                cv::circle(frameWarped, dstPoints[i], 9, color, -1); // Draw filled circle
            }
        }
        else {
            for (const auto& pair : pointsBoundingBoxes) {
                int classID = pair.first;  // The class ID (e.g., 0, 1, 2...)
                const std::vector<cv::Point>& points = pair.second;  // The vector of points for that class

                cv::Scalar color = cv::Scalar(255, 0, 0); // Green color for bounding boxes (you can change it based on classID)

                for (const auto& point : points) {
                    int boxWidth = 25;
                    int boxHeight = 25 ;

                    cv::Point topLeft(point.x - boxWidth / 2, point.y - boxHeight / 2);
                    cv::Point bottomRight(point.x + boxWidth / 2, point.y + boxHeight / 2);

                    // cv::rectangle(frame, topLeft, bottomRight, color, 3);
                    cv::circle(frame, point, 7, cv::Scalar(255, 0, 0),-1);
                    cv::Point textPosition(topLeft.x, topLeft.y - 5);  // Shift text slightly above the top-left point

                    // cv::putText(frame, classMap[classID], textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, color, 3);
                    std::cout << point << std::endl;
                }
            }
            std::cout << " " << std::endl;
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
        const double scaleOutput = configData["output"]["scaleOutput"];
        const int resizeWidth = static_cast<int>(outputWindow.cols*scaleOutput);
        const int resizeHeight = static_cast<int>(outputWindow.rows*scaleOutput);


        // cv::imshow(outputWindowName, frame);
        cv::imshow(outputWindowName, outputWindow);
        // cv::resizeWindow(outputWindowName, resizeWidth, resizeHeight);
        // cv::Point centeredCoords = out.calculateCenteredCoords(resizeWidth, resizeHeight, true);
        // cv::moveWindow(outputWindowName,centeredCoords.x, centeredCoords.y);

        // cv::imshow(outputWindowName, maskInv);

        int key = cv::waitKey(1);
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
