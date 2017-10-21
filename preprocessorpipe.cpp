#include "preprocessorpipe.h"

namespace face {
PreprocessorPipe::PreprocessorPipe()
{
}

void PreprocessorPipe::push_back(Filter& f)
{
    filterPipe.push_back(f);
}

void PreprocessorPipe::insert(uint i, Filter& f)
{
    filterPipe.insert(filterPipe.begin() + i, f);
}

Filter& PreprocessorPipe::filterPipeAt(uint i)
{
    return filterPipe.at(i);
}

void PreprocessorPipe::removeFilter(uint i)
{
    filterPipe.erase(filterPipe.begin() + i);
}

bool PreprocessorPipe::processPipe()
{
    bool result = false;
    for (Filter& filter : filterPipe) {
        filter.setImage4DComponent(imageSet);
        result = filter.filter();
        imageSet = filter.getImage4DComponent();
    }
    return result;
}

vector<Filter> PreprocessorPipe::getFilterPipe() const
{
    return filterPipe;
}

void PreprocessorPipe::setFilterPipe(const vector<Filter>& value)
{
    filterPipe = value;
}

Image4DComponent* PreprocessorPipe::getImageSet() const
{
    return imageSet;
}

void PreprocessorPipe::setImageSet(Image4DComponent* value)
{

    imageSet = value;
}
}
