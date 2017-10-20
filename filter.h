#ifndef FILTER_H
#define FILTER_H
#include <image4d.h>

namespace face {
class Filter {
public:
    virtual bool filter(Image4DSet& image) = 0;
};
}
#endif // FILTER_H
