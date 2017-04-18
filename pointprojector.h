#ifndef POINTPROJECTOR_H
#define POINTPROJECTOR_H
#include <pcl/common/common.h>

/**
 * @brief The PointProjector class projects a point from 3D PointCloud to a 2D image
 */
class PointProjector {

public:
    PointProjector();

    static std::vector<float> get2DCoordinates(pcl::PointXYZ point);
};

#endif // POINTPROJECTOR_H
