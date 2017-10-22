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
    Filter* filterPipeAt(uint i);

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

    vector<Filter*> getFilterPipe() const;
    void setFilterPipe(const vector<Filter*>& value);

    Image4DComponent* getImageSet() const;
    void setImageSet(Image4DComponent* value);

protected:
    vector<Filter*> filterPipe;
    Image4DComponent* imageSet;
};
}
#endif // PREPROCESSORPIPE_H
