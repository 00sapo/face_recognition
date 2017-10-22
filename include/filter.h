#ifndef FILTER_H
#define FILTER_H
#include <image4dleaf.h>

namespace face {
class Filter {
public:
    virtual bool filter() = 0;

    virtual Image4DComponent* getImage4DComponent() const = 0;
    virtual void setImage4DComponent(Image4DComponent* value) = 0;
};
}
#endif // FILTER_H
