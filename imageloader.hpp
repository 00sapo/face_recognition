#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <regex>

#define MATCH_ALL ".*"

/**
 * @brief The ImageLoader class loads a single image or
 *        an image sequence with the possibility of specifying
 *        the filename to load using regular expressions and a
 *        downscaling ratio to reduce image size.
 */
class ImageLoader {
public:
    float downscalingRatio = 1;

    /**
     * @brief ImageLoader, basic constructor.
     *
     * Loads from current directory images with extension .png, .jpg or .bmp
     */
    ImageLoader();

    /**
     * @brief ImageLoader constructor
     * @param dirPath: absolute path to the directory from which load the files
     * @param fileNameTempl: regular expression for the file names to load
     */
    ImageLoader(const std::string& dirPath, const std::string& fileNameRegEx = MATCH_ALL);

    /**
     * @brief hasNext
     * @return true if there are images to load from current directory
     */
    bool hasNext() const;

    /**
     * @brief get
     * @param image: Mat in which store the loaded image
     * @return false if there are no images to load
     */
    bool get(cv::Mat& image);

    /**
     * @brief get
     * @param imageSequence: vector in which store the loaded images
     * @return always true
     *
     * Loads from current directory all files matching the regular expression
     */
    bool get(std::vector<cv::Mat>& imageSequence);

    bool get(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    bool get(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloudSequence);

    /**
     * @brief setFileNameRegEx
     * @param fileNameRegEx: regular expression for the file names to load
     *
     * Changes the regular expression used for file name matching
     */
    void setFileNameRegEx(const std::string& fileNameRegEx);

    /**
     * @brief setCurrentPath
     * @param dirPath: path to a new directory
     *
     * Changes current path to dirPath
     */
    void setCurrentPath(const std::string& dirPath);

private:
    std::string currentPath;
    std::regex fileTemplate;
    std::vector<boost::filesystem::path> fileNames;

    bool loadFileName(const std::string& path);
    bool loadFileNames(const std::string& dirPath);

    bool matchTemplate(const std::string& fileName);
};

#endif // IMAGELOADER_Hs
