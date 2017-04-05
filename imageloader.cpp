#include "imageloader.hpp"

#include <opencv2/imgproc.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

// TODO: this is only a temporay function, it will be moved
//       inside the class once finished
int filesInFolder(string &path) {

    fs::path full_path = fs::system_complete( fs::path( path ) );

    unsigned long file_count = 0;
    unsigned long dir_count = 0;
    unsigned long other_count = 0;
    unsigned long err_count = 0;

    if ( !fs::exists(full_path) ) {
        std::cout << "\nNot found: " << full_path.filename() << std::endl;
        return -1;
    }

    if ( fs::is_directory(full_path) ) {
        std::cout << "\nIn directory: "
                  << full_path.filename() << "\n\n";
        fs::directory_iterator end_iter;
        for ( fs::directory_iterator dir_itr( full_path );
              dir_itr != end_iter;
              ++dir_itr )
        {
            try {
                if ( fs::is_directory( dir_itr->status() ) ) {
                    ++dir_count;
                    std::cout << dir_itr->path().filename() << " [directory]\n";
                }
                else if ( fs::is_regular_file( dir_itr->status() ) )  {
                    ++file_count;
                    std::cout << dir_itr->path().filename() << "\n";
                }
                else {
                    ++other_count;
                    std::cout << dir_itr->path().filename() << " [other]\n";
                }

            }
            catch ( const std::exception &ex ) {
                ++err_count;
                std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
            }
        }
        std::cout << "\n" << file_count << " files\n"
                  << dir_count << " directories\n"
                  << other_count << " others\n"
                  << err_count << " errors\n";
    }
    else // must be a file
    {
        std::cout << "\nFound: " << full_path.filename() << "\n";
    }
    return 0;
}


ImageLoader::ImageLoader() : VideoCapture() {}
ImageLoader::ImageLoader(const string& path) : VideoCapture(path, cv::CAP_IMAGES) {}

bool ImageLoader::get(Mat &image) {
    if(!isOpened())
        return false;

    *this >> image;
    if(image.empty())
        return false;

    if(downscalingRatio != 1) {
        int width = image.cols;
        int height = image.rows;
        resize(image, image, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
    }

    return true;
}

bool ImageLoader::get(vector<Mat> &imageSequence) {

    if(!isOpened())
        return false;

    imageSequence.clear();

    while(true) {
        Mat img;
        *this >> img;
        if(img.empty())
            break;

        if(downscalingRatio != 1) {
            int width = img.cols;
            int height = img.rows;
            resize(img, img, Size(width*downscalingRatio,height*downscalingRatio), INTER_AREA);
        }

        imageSequence.push_back(img);
    }

    return true;
}

bool ImageLoader::get(const std::string &path, cv::Mat &image) {
    open(path, cv::CAP_IMAGES);
    return get(image);
}

bool ImageLoader::get(const std::string &path, std::vector<cv::Mat> &imageSequence) {
    open(path, cv::CAP_IMAGES);
    return get(imageSequence);
}
