#include "Board.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

Board::Board() :
    m_scaleBoard(540.0f/190.0f),
    m_wireThickness(1.0f),
    positionBoard2D(540, 540),
    orientationBoard2D(-6*m_stepAngle+m_stepAngle/2),
    // positionBoard3D(0.0, 173000.0, 237000.0),
    // orientationBoard3D(0.0, 0.0, 0.0),
    m_radiusBullseye(6.35 * m_scaleBoard),
    m_radiusSingleBull(15.9 * m_scaleBoard),
    m_radiusOuterTripleFields(107.0 * m_scaleBoard),
    m_radiusInnerTripleFields((99.0-2.0f*m_wireThickness) * m_scaleBoard),
    m_radiusOuterDoubleFields(170.0 * m_scaleBoard),
    m_radiusInnerDoubleFields((162.0-2.0f*m_wireThickness) * m_scaleBoard),
    m_nrFields(20),
    m_stepAngle(2*M_PI/static_cast<double>(m_nrFields)),
    m_colorLines(0, 255, 0),
    m_colorPoints(255, 0, 0),
    m_colorPointsSingleBull(100, 0, 255),
    m_colorFields(255, 0, 255),
    m_colorStartIndex(0, 255, 0),
    m_radiusPoints(6),
    m_thicknessPoints(-1),
    m_thicknessLines(6),
    m_thicknessFields(m_thicknessLines+1),
    m_enumList({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}),
    m_dartBoardOrder({20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5})

{
    createBoardMapping();
}

void Board::createBoardMapping(){
    std::map <int, int> tempBoardMapping;
    for(int i=0;i<m_dartBoardOrder.size();i++) {
        tempBoardMapping[m_enumList[i]] = m_dartBoardOrder[i];
    }

    m_BoardMapping.assign(tempBoardMapping.begin(), tempBoardMapping.end());
    std::sort(m_BoardMapping.begin(), m_BoardMapping.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    if (a.second == 20) return true;
    if (b.second == 20) return false;
    return a.second < b.second;
    });

}

void Board::calcPoints(const double radius1, const double radius2, std::array<Field, 20>& points) const {

    for(int i=0; i<m_nrFields; i++) {
        const int idx = i % m_nrFields;
        const double angle1 = idx*m_stepAngle;
        const double angle2 = (idx+1)*m_stepAngle;

        const cv::Point p1(static_cast<int>(positionBoard2D.x + radius1 * cos(orientationBoard2D + angle1)),
                     static_cast<int>(positionBoard2D.y + radius1 * sin(orientationBoard2D + angle1)));

        const cv::Point p2(static_cast<int>(positionBoard2D.x + radius1 * cos(orientationBoard2D + angle2)),
                     static_cast<int>(positionBoard2D.y + radius1 * sin(orientationBoard2D + angle2)));

        const cv::Point p3(static_cast<int>(positionBoard2D.x + radius2 * cos(orientationBoard2D + angle1)),
                     static_cast<int>(positionBoard2D.y + radius2 * sin(orientationBoard2D + angle1)));

        const cv::Point p4(static_cast<int>(positionBoard2D.x + radius2 * cos(orientationBoard2D + angle2)),
                     static_cast<int>(positionBoard2D.y + radius2 * sin(orientationBoard2D + angle2)));

        points[i] = {p1,p2,p3,p4};
    }
}

void Board::orderPoints(std::array<Field,20>& points) const {
    std::array<Field,20> pointsOrdered;

    int i=0;
    for (const auto& pair : m_BoardMapping) {
        pointsOrdered[i] = points[pair.first];
        i++;
    }
    points = pointsOrdered;
}

void Board::calcPointsFullBoard() {
    calcPoints(m_radiusOuterDoubleFields, m_radiusInnerDoubleFields, pointsDouble);
    calcPoints(m_radiusInnerDoubleFields, m_radiusOuterTripleFields, pointsOuterSingle);
    calcPoints(m_radiusOuterTripleFields, m_radiusInnerTripleFields, pointsTriple);
    calcPoints(m_radiusInnerTripleFields, m_radiusSingleBull, pointsInnerSingle);
    calcPoints(m_radiusSingleBull, m_radiusBullseye, pointsSingleBull);

    // createBoardMapping();
    orderPoints(pointsTriple);
    orderPoints(pointsDouble);
    orderPoints(pointsInnerSingle);
    orderPoints(pointsOuterSingle);
    orderPoints(pointsSingleBull);
}

void Board::drawPoints(cv::Mat& frame, const std::array<Field, 20>& points) const {
    for(auto& field : points) {
        for(auto& p : field.coordsField) {
            cv::circle(frame, p, m_radiusPoints, m_colorPoints,m_thicknessPoints);
        }
    }
}

void Board::drawPointsFullBoard(cv::Mat& frame) const {
    drawPoints(frame, pointsDouble);
    drawPoints(frame, pointsTriple);
    drawPoints(frame, pointsOuterSingle);
    drawPoints(frame, pointsInnerSingle);
    // drawPoints(frame, pointsSingleBull);
}

void Board::drawLines(cv::Mat& frame) const {
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusBullseye), m_colorLines,m_thicknessLines);
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusSingleBull), m_colorLines,m_thicknessLines);
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusOuterTripleFields), m_colorLines,m_thicknessLines);
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusInnerTripleFields), m_colorLines,m_thicknessLines);
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusOuterDoubleFields), m_colorLines,m_thicknessLines);
    cv::circle(frame, positionBoard2D, static_cast<int>(m_radiusInnerDoubleFields), m_colorLines,m_thicknessLines);

    for(int i=0; i<m_nrFields; i++) {
        const cv::Point center1(pointsSingleBull[i].coordsField[0]);
        const cv::Point stop1(pointsDouble[i].coordsField[0]);
        cv::line(frame, center1, stop1, m_colorLines,m_thicknessLines);

        const cv::Point center2(pointsSingleBull[i].coordsField[1]);
        const cv::Point stop2(pointsDouble[i].coordsField[1]);
        cv::line(frame, center2, stop2, m_colorLines,m_thicknessLines);
    }
    cv::line(frame, pointsSingleBull[0].coordsField[0], pointsDouble[0].coordsField[0], m_colorStartIndex,m_thicknessLines);
}

void Board::drawBoard(cv::Mat &frame) const {
    drawLines(frame);
    // drawPointsFullBoard(frame);
}

void Board::drawField(cv::Mat& frame, const std::string& input) const {

    std::pair<std::string, int> result = extractFieldNr(input);
    std::string field = result.first;
    int nr = result.second;

    if(field == "sb" || field == "db") {
        drawBull(frame,field);
        return;
    }

    if(nr == -1) {
        std::cerr << "Invalid input: " << field <<  nr << std::endl;
        return;
    }

    int radiusOuter;
    int radiusInner;
    std::array<Field, 20> points;
    nr = nr%20;

    if (field == "si") {
        points = pointsInnerSingle;
        radiusOuter = static_cast<int>(m_radiusInnerTripleFields);
        radiusInner = static_cast<int>(m_radiusSingleBull);
    }
    else if (field == "t") {
        points = pointsTriple;
        radiusOuter = static_cast<int>(m_radiusOuterTripleFields);
        radiusInner = static_cast<int>(m_radiusInnerTripleFields);
    }
    else if (field == "so") {
        points = pointsOuterSingle;
        radiusOuter = static_cast<int>(m_radiusInnerDoubleFields);
        radiusInner = static_cast<int>(m_radiusOuterTripleFields);
    }
    else if (field == "d") {
        points = pointsDouble;
        radiusOuter = static_cast<int>(m_radiusOuterDoubleFields);
        radiusInner = static_cast<int>(m_radiusInnerDoubleFields);
    }
    else{
        std::cerr << "Invalid input: " << field <<  nr << std::endl;
        return;
    }

    const Field& cornerPoints = points[nr];
    const int nrStepsFromMapping = m_BoardMapping[nr].first;

    // for(const auto& p : cornerPoints.coordsField) {
    //     cv::circle(frame, p, m_radiusPoints, m_colorFields, m_thicknessPoints);
    // }

    const double startAngle = (orientationBoard2D +nrStepsFromMapping *m_stepAngle) * 180.0 / CV_PI;  // Convert to degrees
    const double endAngle = startAngle + (m_stepAngle * 180.0 / CV_PI);  // Convert to degrees

    cv::ellipse(frame, positionBoard2D, cv::Size(radiusOuter, radiusOuter),
                0, startAngle, endAngle, m_colorFields, m_thicknessFields);

    cv::ellipse(frame, positionBoard2D, cv::Size(radiusInner, radiusInner),
                0, startAngle, endAngle, m_colorFields, m_thicknessFields);

    if (cornerPoints.coordsField.size() == 4) {
        cv::line(frame, cornerPoints.coordsField[0], cornerPoints.coordsField[2], m_colorFields, m_thicknessFields);
        cv::line(frame, cornerPoints.coordsField[1], cornerPoints.coordsField[3], m_colorFields, m_thicknessFields);
    }
}

void Board::drawBull(cv::Mat& frame, const std::string& field) const {
    int radius = 0;
    if (field == "sb") {
        radius = static_cast<int>(m_radiusSingleBull);
    }
    else if (field == "db") {
        radius = static_cast<int>(m_radiusBullseye);
    }
    cv::circle(frame, positionBoard2D, radius, m_colorFields, m_thicknessFields);
}

std::pair<std::string, int> Board::extractFieldNr(const std::string& input) {
    std::string field;
    int nr = -1;
    size_t pos = input.find_first_of("0123456789");
    if(pos!=std::string::npos) {
        field = input.substr(0,pos);
        std::string nrString = input.substr(pos);
        std::istringstream(nrString) >> nr;
    }
    else {
        field = input;
    }
    return std::make_pair(field, nr);
}

std::array<cv::Point, 4> Board::getPointsFromField(std::string field, const int index) {
    if(field == "d")
        return pointsDouble[index].coordsField;
    else if(field == "t")
        return pointsTriple[index].coordsField;
    else if(field == "si")
        return pointsInnerSingle[index].coordsField;
    else if(field == "so")
        return pointsOuterSingle[index].coordsField;
    else if(field == "sb")
        return pointsSingleBull[index].coordsField;
}

cv::Point Board::getPositionBullseye() {
    return positionBoard2D;
}

