#ifndef PREPROCESSORPIPE_H
#define PREPROCESSORPIPE_H
#include <filter.h>
#include <vector>

using std::vector;

namespace face {
class PreprocessorPipe {
public:
    PreprocessorPipe();

    /**
     * @brief push_back add filter f at the end of the pipe
     * @param f
     */
    void push_back(Filter& f);

    /**
     * @brief insert insert a Filter at position i in the pipe
     * @param i
     * @param f
     */
    void insert(uint i, Filter& f);

    /**
     * @brief at
     * @param i
     * @return a pointer to the Filter at position i in the pipe
     */
    Filter& filterPipeAt(uint i);

    /**
     * @brief removeFilter remove filter
     * at index i in the pipe. Use std::vector::erase(), it is slow.
     * @param i
     */
    void removeFilter(uint i);

    /**
     * @brief processPipe apply filter pipe to the image set
     * @return true if all filters managed to work, false otherwise
     */
    bool processPipe();

    /**
     * @brief push_back add the Image4DSetComponent to the image set
     * @param image4d
     */
    void push_back(Image4DSetComponent& image4d);

    /**
     * @brief imageSetAt
     * @param i
     * @return the Image4DSetComponent at position i
     */
    Image4DSetComponent& imageSetAt(uint i);

    vector<Filter> getFilterPipe() const;
    void setFilterPipe(const vector<Filter>& value);

    vector<Image4DSetComponent> getImageSet() const;
    void setImageSet(const vector<Image4DSetComponent>& value);

protected:
    vector<Filter> filterPipe;
    vector<Image4DSetComponent> imageSet;
};
}
#endif // PREPROCESSORPIPE_H
