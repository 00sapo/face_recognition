#ifndef FACECROPPER_H
#define FACECROPPER_H
#include "extern_libs/head_pose_estimation/CRForestEstimator.h"
#include <filter.h>
#include <opencv2/core.hpp>

using cv::Vec3f;

namespace face {
class FaceCropper : public Filter {
public:
    FaceCropper();

    bool filter(Image4DSet& image4d);

private:
    void removeOutliers(Image4DSet& image4d) const;
    bool estimateFacePose(const Image4D& image4d, cv::Vec3f& position, cv::Vec3f& eulerAngles) const;

    CRForestEstimator estimator;
    bool poseEstimatorAvailable = false;
};
}
#endif // FACECROPPER_H
