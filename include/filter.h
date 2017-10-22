#ifndef FILTER_H
#define FILTER_H
#include <image4dleaf.h>

namespace face {
class Filter {
public:
    virtual bool filter() = 0;

    virtual Image4DComponent* getImage4DComponent() const;
    virtual void setImage4DComponent(Image4DComponent* value);
};
}
#endif // FILTER_H
