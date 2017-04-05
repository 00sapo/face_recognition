#include "yamlloader.h"
#include <iostream>

using namespace std;

void testYaml()
{
    YamlLoader loader = YamlLoader();

    //    loader.setPath("../example.yaml");
    loader.setPath("camera_info.yaml");

    loader.read();

    SingletonSettings* settings = SingletonSettings::getInstance();
    cout << settings->getD() << endl
         << settings->getK() << endl
         << settings->getP() << endl
         << settings->getR() << endl
         << settings->getHeight() << endl
         << settings->getWidth() << endl;
}

int main()
{
    cout << "Hello World!" << endl;
    testYaml();

    return 0;
}
