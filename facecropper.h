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

    bool filter();

    Image4DSetComponent* getImage4d() const;
    void setImage4d(Image4DSetComponent* value);

    CRForestEstimator getEstimator() const;
    void setEstimator(const CRForestEstimator& value);

    bool isPoseEstimatorAvailable() const;

private:
    void removeOutliers() const;
    bool estimateFacePose() const;

    Image4DSetComponent* image4d;

    CRForestEstimator estimator;
    bool poseEstimatorAvailable = false;
};
}
#endif // FACECROPPER_H
