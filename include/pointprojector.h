#ifndef POINTPROJECTOR_H
#define POINTPROJECTOR_H
#include <pcl/common/common.h>

/**
 * @brief The PointProjector class projects a point from 3D PointCloud to a 2D image
 */
class PointProjector {

public:
    PointProjector();

    /**
     * @brief get2DCoordinates: returns the coordinates of 3D point projected in a 2D space
     * @param point
     * @return
     */
    static std::vector<unsigned int> get2DCoordinates(pcl::PointXYZ point);
};

#endif // POINTPROJECTOR_H
