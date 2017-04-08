#ifndef BACKGROUNDSEGMENTATION_H
#define BACKGROUNDSEGMENTATION_H
#include <opencv2/opencv.hpp>

/**
 * @brief The BackgroundSegmentation class performs the removing of background from a RGB-D image.
 */
class BackgroundSegmentation {
public:
    BackgroundSegmentation();

    /**
     * @brief setImageCollection: set the collection of images to be filter
     * @param imageSequence: A vector containing the collection (i.e. the one given by ImageLoader)
     */
    void setImageCollection(std::vector<cv::Mat>& imageSequence);

    /**
     * @brief findTreshold: Automatically find the threshold to be used for filtering
     * @return A double containg the threshold value
     */
    double findTreshold();

    /**
     * @brief filter: Remove the background in each image of the collection using the threshold
     * @return return True if everything worked, false otherwise.
     */
    bool filter();

    /**
     * @brief filter: set the threshold and calls filter()
     * @param threshold: the value to use as threshold
     * @return same as filter
     */
    bool filter(double threshold);

    double getTreshold() const;
    void setTreshold(double value);

private:
    std::vector<cv::Mat> imageCollection;
    double treshold;
};

#endif // BACKGROUNDSEGMENTATION_H
