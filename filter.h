#ifndef FILTER_H
#define FILTER_H
#include <image4dleaf.h>

namespace face {
class Filter {
public:
    virtual bool filter() = 0;

    virtual Image4DSetComponent* getImage4d() const;
    virtual void setImage4d(Image4DSetComponent* value);
};
}
#endif // FILTER_H
