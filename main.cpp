#include <iostream>

#include <opencv2/highgui.hpp>

#include <pcl/visualization/area_picking_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "backgroundsegmentation.h"
#include "face.h"
#include "faceloader.h"
#include "singletonsettings.h"
#include "yamlloader.h"

using namespace std;
using namespace cv;
using namespace pcl;

void testYaml()
{
    YamlLoader loader = YamlLoader();

    loader.setPath("camera_info.yaml");

    loader.read();

    SingletonSettings& settings = SingletonSettings::getInstance();
    cout << settings.getD() << endl
         << settings.getK() << endl
         << settings.getP() << endl
         << settings.getR() << endl
         << settings.getHeight() << endl
         << settings.getWidth() << endl;
}

void keyboardEventHandler(const visualization::KeyboardEvent& event, void* viewer_void)
{

    //    boost::shared_ptr<visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<visualization::PCLVisualizer>*>(viewer_void);
    visualization::PCLVisualizer* viewer = (visualization::PCLVisualizer*)viewer_void;

    if (event.getKeySym() == "n" && event.keyDown())
        viewer->close();
}

void testFaceLoader()
{
    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*");
    vector<Face> faceSequence(0);
    if (!loader.get(faceSequence)) {
        cout << "Error loading face!" << endl;
        return;
    }
    cout << "\n\nFaces loaded!" << endl;

    namedWindow("image", WINDOW_NORMAL);
    for (const auto& face : faceSequence) {
        imshow("image", face.image);
        while (waitKey(0) != 'm') {
        }

        visualization::PCLVisualizer* viewer = new visualization::PCLVisualizer("PCL Viewer");
        viewer->setBackgroundColor(0.0, 0.0, 0.5);
        viewer->addCoordinateSystem(0.1);
        viewer->initCameraParameters();
        //visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
        viewer->addPointCloud<PointXYZ>(face.cloud, "input_cloud");

        viewer->registerKeyboardCallback(keyboardEventHandler, (void*)viewer);
        while (!viewer->wasStopped()) {
            viewer->spin();
        }
    }
}

void testFindThreshold()
{
    BackgroundSegmentation segmenter;

    string dirPath = "../RGBD_Face_dataset_training/";
    FaceLoader loader(dirPath, "014.*"); // example: loads only .png files starting with 014

    Face face;

    if (!loader.get(face)) {
        cout << "Failed loading face" << endl;
        return;
    }

    cout << "Face loaded!" << endl;

    cout << "\nFiltering background..." << endl;

    segmenter.filterBackground(face);

    //cout << "Treshold found: " << segmenter.findTreshold() << endl;
}

int main()
{
    cout << "Hello World!" << endl;

    cout << "Yaml test..." << endl;
    testYaml();

    cout << "\n\nFace loader test..." << endl;
    testFaceLoader();

    cout << "\n\nFind threshold test..." << endl;
    testFindThreshold();

    cout << "\n\nTests finished!" << endl;

    //    testCloudLoader();
    return 0;
}
