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
    for (Image4DSetComponent& image4d : imageSet) {
        for (Filter& filter : filterPipe) {
            filter.setImage4d(&image4d);
            result = filter.filter();
        }
    }
    return result;
}

void PreprocessorPipe::push_back(Image4DSetComponent& image4d)
{
    imageSet.push_back(image4d);
}

Image4DSetComponent& PreprocessorPipe::imageSetAt(uint i)
{
    return imageSet.at(i);
}

vector<Filter> PreprocessorPipe::getFilterPipe() const
{
    return filterPipe;
}

void PreprocessorPipe::setFilterPipe(const vector<Filter>& value)
{
    filterPipe = value;
}

vector<Image4DSetComponent> PreprocessorPipe::getImageSet() const
{
    return imageSet;
}

void PreprocessorPipe::setImageSet(const vector<Image4DSetComponent>& value)
{
    imageSet = value;
}
}
