#ifndef CLUSTEROFIMAGES_H
#define CLUSTEROFIMAGES_H
#include <image4dleaf.h>
#include <image4dsetcomponent.h>
#include <vector>

using std::vector;

namespace face {
class Image4DClusterComposite : public Image4DSetComponent {
public:
    Image4DClusterComposite();

private:
    vector<Image4DLeaf> cluster;
};
}
#endif // CLUSTEROFIMAGES_H
