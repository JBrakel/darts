#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include <nlohmann/json.hpp>

class Config {
public:
    nlohmann::json importDataJson(const std::string& filename);
};



#endif //CONFIG_H
