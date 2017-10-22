#include "poseclusterizer.h"

using std::vector;

namespace face {
PoseClusterizer::PoseClusterizer(int numCenters)
    : numCenters(numCenters)
{
}

bool PoseClusterizer::clusterizePoses()
{
    if (imageSet->size() < numCenters)
        return false;

    // retrieve faces' poses
    vector<Pose> poses = imageSet->getRotationMatrix();

    // clusterize poses
    vector<int> bestLabels;
    cv::Mat centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS, 10, 1.0);
    cv::kmeans(poses, numCenters, bestLabels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.rows != numCenters) {
        std::cout << "Clustering poses failed!" << std::endl;
        this->centers = vector<Pose>(numCenters);
        return false; // TODO: check default Pose value
    }

    // convert from Mat to vector<Pose>
    vector<Pose> ctrs;
    for (int i = 0; i < centers.rows; ++i) {
        ctrs.emplace_back(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2),
            centers.at<float>(i, 3), centers.at<float>(i, 4), centers.at<float>(i, 5),
            centers.at<float>(i, 6), centers.at<float>(i, 7), centers.at<float>(i, 8));
    }
    this->centers = ctrs;
    return true;
}

int PoseClusterizer::getNearestCenterId(const Pose& pose)
{
    float min = std::numeric_limits<float>::max();
    int index = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        float norm = cv::norm(centers[i], pose, cv::NORM_L2);
        if (norm < min) {
            min = norm;
            index = i;
        }
    }
    return index;
}

void PoseClusterizer::assignFacesToClusters()
{
    vector<Image4DComponent> clusters(numCenters);
    for (auto& image : *imageSet) {
        if (image.isLeaf()) {
            int index = getNearestCenterId(image.getRotationMatrix()[0]);
            clusters.at(index).add(image);
        } else {
            //It should ever eter this block, I inserted it for completeness --sapo
            PoseClusterizer pc;
            pc.setImage4DComponent(imageSet);
            pc.assignFacesToClusters();

            for (int i = 0; i < numCenters; i++)
                clusters[i].add(*pc.getImage4DComponent()->at(i));
        }
    }

    imageSet->clear();

    for (auto& component : clusters) {
        imageSet->add(component);
    }
}

bool PoseClusterizer::filter()
{
    bool result = clusterizePoses();
    if (result)
        assignFacesToClusters();
    return result;
}

Image4DComponent* PoseClusterizer::getImage4DComponent() const
{
    return imageSet;
}

void PoseClusterizer::setImage4DComponent(Image4DComponent* value)
{
    imageSet = value;
}

int PoseClusterizer::getNumCenters() const
{
    return numCenters;
}

void PoseClusterizer::setNumCenters(int value)
{
    numCenters = value;
}

std::vector<Pose> PoseClusterizer::getCenters() const
{
    return centers;
}

void PoseClusterizer::setCenters(const std::vector<Pose>& value)
{
    centers = value;
}
}
