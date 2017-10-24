#ifndef COVARIANCECOMPUTER_H
#define COVARIANCECOMPUTER_H
#include <filter.h>
#include <image4dcomponent.h>
#include <opencv2/core.hpp>
#include <vector>

namespace face {
/**
     * @brief computeCovarianceRepresentation: extracts a normalized covariance matrix based representation
     *        of an input Face set. Input faces are clusterized in subsets based on their pose
     *        and then for each subset a pair of covariance matrixes, one for images and the other
     *        for the depth maps, representative of the set are computed. If this is a leaf
     *        covariance computer, it computes covariances for single leaves, otherwise it does
     *        not compute covariance for the whole Image4DComponent setted in this Filter, but only
     *        for its subcomponents in the second-last level.
     */
class CovarianceComputer : public Filter {
public:
    std::string actionToPrint() { return "Computing covariances..."; }
    CovarianceComputer();

    bool filter();
    Image4DComponent* getImage4DComponent() const;
    void setImage4DComponent(Image4DComponent* value);

    bool isLeafCovarianceComputer();

    void setLeafCovarianceComputer(bool value);

private:
    /**
     * @brief setToCovariance: computes the covariance matrix representation of an Image4DComponent.
     *                         It is thinked to act on leaves or on components that contains only leaves.
     */
    bool setToNormalizedCovariance(Image4DComponent& set);
    bool leafCovarianceComputer = false;
    Image4DComponent* imageSet;
};
}
#endif // COVARIANCECOMPUTER_H
