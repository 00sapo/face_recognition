#ifndef CLUSTEROFIMAGES_H
#define CLUSTEROFIMAGES_H
#include <image4d.h>
#include <imageset.h>
#include <vector>

using std::vector;

namespace face {
class ClusterOfImages4D : public Image4DSet {
public:
    ClusterOfImages4D();

private:
    vector<Image4D> cluster;
};
}
#endif // CLUSTEROFIMAGES_H
