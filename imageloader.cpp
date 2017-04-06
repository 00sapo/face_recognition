#include "imageloader.hpp"

#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

ImageLoader::ImageLoader() {
    currentPath = fs::current_path().string();
    fileTemplate = regex(".*(png|jpg|bmp)");
    fileNames = vector<string>();
}

ImageLoader::ImageLoader(const string& dirPath, const string &fileNameRegEx) {
    fileNames = vector<string>();
    currentPath = dirPath;
    fileTemplate = regex(fileNameRegEx);
    loadFileNames(currentPath);
}

bool ImageLoader::hasNext() const {
    return !fileNames.empty();
}

bool ImageLoader::get(Mat &image) {

    if(!hasNext())
        return false;

    string file = fileNames.back();
    cout << "Loading " << file << " file" << endl;
    image = cv::imread(fileNames.back());
    fileNames.pop_back();

    if(downscalingRatio != 1) {
        int width = image.cols;
        int height = image.rows;
        resize(image, image, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
    }
    return true;
}

bool ImageLoader::get(vector<Mat> &imageSequence) {

    imageSequence.clear();

    while(hasNext()) {
        Mat img;
        get(img);
        imageSequence.push_back(img);
    }

    return true;
}

/*
bool ImageLoader::get(const std::string &path, cv::Mat &image) {
    loadFileName(path);
    return get(image);
}

bool ImageLoader::get(const std::string &dirPath, std::vector<cv::Mat> &imageSequence) {
    loadFileNames(dirPath);
    return get(imageSequence);
}
*/


void ImageLoader::setFileNameRegEx(const std::string& fileNameRegEx) {
    fileTemplate = regex(fileNameRegEx);
    loadFileNames(currentPath);
}

void ImageLoader::setCurrentPath(const string &dirPath)
{
    fileNames.clear();
    currentPath = dirPath;
    loadFileNames(currentPath);
}

bool ImageLoader::loadFileName(const string &path) {
    fs::path full_path = fs::system_complete(fs::path(path));

    // check if exsists
    if (!fs::exists(full_path)) {
        cerr << "\nNot found: " << full_path.filename() << endl;
        return false;
    }

    // adds to the LIFO list of files to load
    fileNames.push_back(path);

    return true;
}

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
                if (matchTemplate(iter->path().filename().string())) {
                    fileNames.push_back(iter->path().string());
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
