#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include <regex>
#include <mutex>

class Image4D;

/**
 * @brief The ImageLoader class loads a single image or
 *        an image sequence with the possibility of specifying
 *        the filename to load using regular expressions and a
 *        downscaling ratio to reduce image size.
 */
class Image4DLoader {
public:

    static const std::string MATCH_ALL;

    /**
     * @brief ImageLoader, basic constructor.
     *
     * Loads from current directory images with extension .png, .jpg or .bmp
     */
    Image4DLoader();

    /**
     * @brief ImageLoader constructor
     * @param dirPath: absolute path to the directory from which load the files
     * @param fileNameTempl: regular expression for the file names to load
     */
    Image4DLoader(const std::string& dirPath, const std::string& fileNameRegEx = MATCH_ALL);

    /**
     * @brief hasNext
     * @return true if there are images to load from current directory
     */
    bool hasNext() const;

    /**
     * @brief get: loads the next image and cloud files available
     * @param image4d: output image4d
     * @return true if successfully loaded, false otherwise
     */
    bool get(Image4D& image4d);

    /**
     * @brief get: multithreaded version of get(Image4D& image4d).
     *        Loads all files using a variable number of threads
     *        depending on the number of available cores
     * @return loaded files
     */
    std::vector<Image4D> get();

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

    float getLeafSize() const;
    void setLeafSize(float value);

private:
    /**
     * @brief leafSize: not working for now. If it is setted, the images will be filtered with a Voxel Grid filter of this leaf size.
     *
     */
    float leafSize = 0.0f;

    std::string currentPath;
    std::regex fileTemplate;
    std::vector<std::string> imageFileNames;
    std::vector<std::string> cloudFileNames;

    bool loadFileNames(const std::string& dirPath);

    bool matchTemplate(const std::string& fileName);

    void getMultiThr(std::vector<Image4D> &image4DSequence, int begin, int end, std::mutex &mutex) const;

};


#endif // IMAGELOADER_Hs
