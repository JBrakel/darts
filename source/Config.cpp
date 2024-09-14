#include "Config.h"
#include <fstream>
#include <iostream>


nlohmann::json Config::importDataJson(const std::string& filename) {
    std::ifstream f(filename);
    nlohmann::json data = nlohmann::json::parse(f);
    return data;
}












