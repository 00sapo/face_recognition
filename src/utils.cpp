#include "utils.h"

#include <iostream>
#include <memory>

#include <opencv2/highgui.hpp>

/*
using pcl::PointXYZ;
using pcl::PointCloud;
namespace visual = pcl::visualization;

void keyboardEventHandler(const visual::KeyboardEvent& event, void* viewer_void)
{

    // boost::shared_ptr<visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<visualization::PCLVisualizer>*>(viewer_void);
    visual::PCLVisualizer* viewer = static_cast<visual::PCLVisualizer*>(viewer_void);

    if (event.getKeySym() == "n" && event.keyDown())
        viewer->close();
}

void viewPointCloud(PointCloud<PointXYZ>::Ptr cloud)
{

    auto viewer = std::make_unique<visual::PCLVisualizer>("PCL Viewer");
    viewer->setBackgroundColor(0.0, 0.0, 0.5);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();

    //visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<PointXYZ>(cloud, "input_cloud");
    viewer->setCameraPosition(-0.24917, -0.0187087, -1.29032, 0.0228136, -0.996651, 0.0785278);

    viewer->registerKeyboardCallback(keyboardEventHandler, viewer.get());
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

void viewPointCloud(const cv::Mat& depthMap) {
    cv::Mat depth;
    //cv::normalize(depthMap, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Depth Map", depthMap);
    std::cout << "Press any key to continue." << std::endl;
    cv::waitKey(0);

}
*/
