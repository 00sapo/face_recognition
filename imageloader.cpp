#include "imageloader.hpp"

#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>

#include "face.h"

using namespace std;
using namespace cv;
using namespace pcl;

namespace fs = boost::filesystem;

ImageLoader::ImageLoader() {
    currentPath = fs::current_path().string();
    fileTemplate = regex(".*(png|jpg|bmp)");
    imageFileNames = vector<fs::path>();
}

ImageLoader::ImageLoader(const string& dirPath, const string &fileNameRegEx) {
    imageFileNames = vector<fs::path>();
    currentPath = dirPath;
    fileTemplate = regex(fileNameRegEx);
    loadFileNames(currentPath);
}

bool ImageLoader::hasNext() const {
    return !imageFileNames.empty();
}

bool ImageLoader::get(Mat &image) {

    if(!hasNext())
        return false;

    fs::path file = imageFileNames.back();
    cout << "Loading " << file.filename() << " file" << endl;
    image = cv::imread(file.filename().string());
    imageFileNames.pop_back();

    if(downscalingRatio != 1) {
        int width = image.cols;
        int height = image.rows;
        resize(image, image, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
    }
    return true;
}

bool ImageLoader::get(vector<Mat>& imageSequence) {

    imageSequence.clear();

    while(hasNext()) {
        Mat img;
        if(!get(img))
            return false;
        imageSequence.push_back(img);
    }

    return true;
}

bool ImageLoader::get(PointCloud<PointXYZ>::Ptr& cloud) {

    if(!hasNext())
        return false;

    fs::path file = imageFileNames.back();
    cout << "Loading" << file.filename() << " file" << endl;
    int result = pcl::io::loadPCDFile<pcl::PointXYZ> (file.filename().string(), *cloud);
    imageFileNames.pop_back();

    return result != -1;

}

bool ImageLoader::get(vector<PointCloud<PointXYZ>::Ptr>& cloudSequence) {

    cloudSequence.clear();

    while(hasNext()) {
        PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);
        if(!get(cloud))
            return false;
        cloudSequence.push_back(cloud);
    }

    return true;
}

bool ImageLoader::get(Face& face) {

    if(!hasNext())
        return false;

    string imageFile = imageFileNames.back();
    string cloudFile = cloudFileNames.back();

    if (imageFile.compare(cloudFile) != 0)
        return false;

    imageFileNames.pop_back();
    cloudFileNames.pop_back();

    cout << "Loading image " << file.filename() << " file" << endl;
    face.imageRGB = cv::imread(file.filename().string());

    if(downscalingRatio != 1) {
        int width = face.imageRGB.cols;
        int height = face.imageRGB.rows;
        resize(face.imageRGB, face.imageRGB, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
    }

    int result = pcl::io::loadPCDFile<pcl::PointXYZ> (cloudFile, face.imageD);

    return true;
}

void ImageLoader::setFileNameRegEx(const std::string& fileNameRegEx) {
    fileTemplate = regex(fileNameRegEx);
    loadFileNames(currentPath);
}

void ImageLoader::setCurrentPath(const string &dirPath)
{
    imageFileNames.clear();
    currentPath = dirPath;
    loadFileNames(currentPath);
}

//bool ImageLoader::loadFileName(const string &path) {
//    fs::path full_path = fs::system_complete(fs::path(path));
//
//    // check if exsists
//    if (!fs::exists(full_path)) {
//        cerr << "\nNot found: " << full_path.filename() << endl;
//        return false;
//    }
//
//    // adds to the LIFO list of files to load
//    fileNames.push_back(full_path);
//
//    return true;
//}

bool ImageLoader::loadFileNames(const string &dirPath)
{
    fs::path full_path = fs::system_complete(fs::path(dirPath));

    // check if exsists
    if (!fs::exists(full_path)) {
        cerr << "\nNot found: " << full_path.filename() << endl;
        return false;
    }

    // check if directory
    if ( !fs::is_directory(full_path) ) {
        cerr << "\n" << full_path.filename() << " is not a directory" << endl;
        return false;
    }

    // iterate trough files and save
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(full_path); iter != end_iter; ++iter) {
        try {
            if (fs::is_regular_file(iter->status()))  {
                cout << "Trying to match: " << iter->path() << endl;
                if (matchTemplate(iter->path().stem().string())) {
                    if(iter->path().extension().string().compare(".png"))
                        imageFileNames.push_back(iter->path().string());
                    if(iter->path().extension().string().compare(".pcd"))
                        cloudFileNames.push_back(iter->path().string());
                }
                //cout << iter->path().filename() << "\n";
            }
        }
        catch ( const std::exception &ex ) {
            cout << iter->path().filename() << " " << ex.what() << endl;
            return false;
        }
    }



    return true;
}


bool ImageLoader::matchTemplate(const string &fileName) {
    return regex_match(fileName,fileTemplate, regex_constants::match_any);
}
