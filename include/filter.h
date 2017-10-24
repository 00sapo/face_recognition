#ifndef FILTER_H
#define FILTER_H
#include <image4dleaf.h>

namespace face {
class Filter {
public:
    /**
     * @brief actionToPrint is a string intended to be printed to describe the filter
     */
    virtual std::string actionToPrint() = 0;
    virtual bool filter() = 0;

    virtual Image4DComponent* getImage4DComponent() const = 0;
    virtual void setImage4DComponent(Image4DComponent* value) = 0;
};
}
#endif // FILTER_H
