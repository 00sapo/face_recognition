#include "svmmanager.h"
#include "covariancecomputer.h"

using namespace face;

SVMManager::SVMManager(int c)
{
    this->c = c;
}

vector<string> SVMManager::generateLabels(int numOfLabels)
{
    string id = "identity_";
    vector<string> identities;

    int numOfDigits = 0;
    for (int N = numOfLabels; N > 0; N /= 10, ++numOfDigits)
        ; // count number of digits

    for (int i = 0; i < numOfLabels; ++i) {
        std::stringstream stream;
        stream << id << std::setfill('0') << std::setw(numOfDigits) << i; // fixed length identity
        identities.push_back(stream.str());
    }

    return identities;
}
