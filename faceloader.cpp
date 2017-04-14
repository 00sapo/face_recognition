#include "faceloader.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <pcl/io/pcd_io.h>

#include "face.h"

using namespace std;
using namespace cv;
using namespace pcl;

namespace fs = boost::filesystem;

FaceLoader::FaceLoader() {
    currentPath = fs::current_path().string();
    fileTemplate = regex(".*(png|jpg|bmp)");
    imageFileNames = vector<string>();
    cloudFileNames = vector<string>();
}

FaceLoader::FaceLoader(const string& dirPath, const string &fileNameRegEx) {
    imageFileNames = vector<string>();
    cloudFileNames = vector<string>();
    currentPath = dirPath;
    fileTemplate = regex(fileNameRegEx);

    cout << "FaceLoader constructor: loading file names..." << endl;
    if(!loadFileNames(currentPath))
        cout << "Failed!" << endl;
}

bool FaceLoader::hasNext() const {
    return !imageFileNames.empty() && !cloudFileNames.empty();
}

bool FaceLoader::get(Face& face) {

    if(!hasNext())
        return false;

    string& imageFile = imageFileNames.back();
    string& cloudFile = cloudFileNames.back();

    //if (imageFile.compare(cloudFile) != 0) {
    //    cerr << "Error! Different filenames" << endl;
    //   return false;
    //}

    face.image = cv::imread(imageFile);
    if(face.image.empty()) {
        cout << "Unable to load file " << imageFile << endl;
        return false;
    }

    // TODO: how should this work with point clouds?
    //if(downscalingRatio != 1) {
    //    int width = face.imageRGB.cols;
    //    int height = face.imageRGB.rows;
    //    resize(face.imageRGB, face.imageRGB, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
    //}

    int result = pcl::io::loadPCDFile<pcl::PointXYZ> (cloudFile, *(face.cloud));
    if(result == -1) {
        cout << "Unable to load file " << cloudFile << endl;
        return false;
    }

    imageFileNames.pop_back();
    cloudFileNames.pop_back();

    return true;
}

bool FaceLoader::get(vector<Face>& faceSequence) {

    faceSequence.clear();
    faceSequence.reserve(imageFileNames.size());
    while(hasNext()) {
        faceSequence.emplace_back();
        if(!get(faceSequence.back()))
            return false;
    }

    return true;

}

void FaceLoader::setFileNameRegEx(const string& fileNameRegEx) {
    fileTemplate = regex(fileNameRegEx);
    loadFileNames(currentPath);
}

void FaceLoader::setCurrentPath(const string& dirPath)
{
    imageFileNames.clear();
    cloudFileNames.clear();
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

bool FaceLoader::loadFileNames(const string &dirPath)
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
    fs::directory_iterator iter(full_path);
    for (auto& dir_entry : iter) {
        try {
            if (fs::is_regular_file(dir_entry.status()))  {
                const fs::path& path = dir_entry.path();
                if (matchTemplate(path.stem().string())) {
                    if(path.extension().string().compare(".png") == 0)
                        imageFileNames.push_back(path.string());
                    else if(path.extension().string().compare(".pcd") == 0)
                        cloudFileNames.push_back(path.string());
                }
            }
        }
        catch ( const std::exception &ex ) {
            cerr << dir_entry.path().filename() << " " << ex.what() << endl;
            return false;
        }
    }

    std::sort(imageFileNames.begin(), imageFileNames.end());
    std::sort(cloudFileNames.begin(), cloudFileNames.end());

    for (const auto& s : imageFileNames) {
        cout << s << endl;
    }
    for (const auto& s : cloudFileNames) {
        cout << s << endl;
    }

    return true;
}


bool FaceLoader::matchTemplate(const string &fileName) {
    return regex_match(fileName,fileTemplate, regex_constants::match_any);
}
