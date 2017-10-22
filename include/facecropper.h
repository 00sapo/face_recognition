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

    Image4DComponent* getImage4DComponent() const;
    void setImage4DComponent(Image4DComponent* value);

    CRForestEstimator getEstimator() const;
    void setEstimator(const CRForestEstimator& value);

    bool isPoseEstimatorAvailable() const;

private:
    void removeOutliers() const;
    bool estimateFacePose() const;

    Image4DComponent* image4d;

    CRForestEstimator estimator;
    bool poseEstimatorAvailable = false;
    static void filterImage4DComponent(Image4DComponent* image4d);
};
}
#endif // FACECROPPER_H
