#ifndef BOARD_H
#define BOARD_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

class Board {
private:
    const double m_scaleBoard;
    const double m_wireThickness;
    const double m_radiusBullseye;
    const double m_radiusSingleBull;
    const double m_radiusOuterTripleFields;
    const double m_radiusInnerTripleFields;
    const double m_radiusOuterDoubleFields;
    const double m_radiusInnerDoubleFields;
    const int m_nrFields;
    const double m_stepAngle;
    const cv::Scalar m_colorLines;
    const cv::Scalar m_colorPoints;
    const cv::Scalar m_colorPointsSingleBull;
    const cv::Scalar m_colorFields;
    const cv::Scalar m_colorStartIndex;
    const int m_radiusPoints;
    const int m_thicknessPoints;
    const int m_thicknessLines;
    const int m_thicknessFields;
    std::array<int, 20> m_dartBoardOrder;
    std::vector<std::pair<int, int>> m_BoardMapping;
    std::array<int,20> m_enumList;

public:
    Board();

    cv::Point positionBoard2D;
    double orientationBoard2D;
    cv::Point3f positionBoard3D;
    cv::Point3f orientationBoard3D;

    struct Field {
        std::array<cv::Point, 4> coordsField;
    };

    std::array<Field, 20> pointsSingleBull;
    std::array<Field, 20> pointsInnerSingle;
    std::array<Field, 20> pointsOuterSingle;
    std::array<Field, 20> pointsDouble;
    std::array<Field, 20> pointsTriple;

    std::array<Field, 20> pointsDoubleTranslated;

    void calcPoints(double radius1, double radius2, std::array<Field, 20>& points) const;
    void createBoardMapping();
    void orderPoints(std::array<Field,20>& points) const;
    void calcPointsFullBoard();
    void drawPoints(cv::Mat& frame, const std::array<Field, 20>& points) const;
    void drawPointsFullBoard(cv::Mat& frame) const;
    void drawLines(cv::Mat& frame) const;
    void drawBoard(cv::Mat& frame) const ;
    static std::pair<std::string, int> extractFieldNr(const std::string& input) ;
    void drawField(cv::Mat& frame, const std::string& input) const;
    void drawBull(cv::Mat& frame, const std::string& field) const;

    std::array<cv::Point, 4> getPointsFromField(std::string field, int index);
    cv::Point getPositionBullseye();
};



#endif //BOARD_H
