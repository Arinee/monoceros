#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include "gflags/gflags.h"

DEFINE_string(data, "b.dat", "build data for input");
DEFINE_uint64(thd, 100000, "threshold for flat or ivf");
DEFINE_bool(dry, false, "dry run for checking statistics");

void tokenize(std::string const &str, const char delim,
                std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::ifstream sIn(FLAGS_data);
    if (!sIn) {
        std::cerr << "Fail to open input file: " << FLAGS_data << std::endl;
        return 0;
    }

    size_t thd = FLAGS_thd;
    bool dry = FLAGS_dry;

    auto start = std::chrono::steady_clock::now();
    std::cout << "Processing..." << std::endl;

    using g_t = size_t;
    using c_t = size_t;
    using d_t = std::string;
    std::map<c_t, std::vector<g_t>> cgMap;
    std::map<g_t, d_t> gdMap;

    std::string lineStr;
    std::vector<std::string> lineVec;
    while(std::getline(sIn, lineStr)) {
        lineVec.clear();
        tokenize(lineStr, '|', lineVec);
        if (lineVec.size() != 3) {
            std::cerr << "Invalid line: " << lineStr << std::endl;
            lineStr.clear();
            lineVec.clear();
            continue;//return -1;
        }
        cgMap[std::stoul(lineVec[0])].push_back(std::stoul(lineVec[1]));
        if (gdMap.count(std::stoul(lineVec[1])) != 0) {
            std::cerr << "[ERROR] Duplicated good id exists: " << lineVec[1] << std::endl;
            std::cerr << "[ERROR] Aborting now..." << std::endl;
            return -1;
        }
        gdMap[std::stoul(lineVec[1])] = lineVec[2];
    }


    std::cout << "CatNum: " << cgMap.size() << "   GoodNum: " << gdMap.size() << std::endl;

    std::string flat(FLAGS_data), ivf(FLAGS_data);
    flat += ".flat";
    ivf += ".ivf";
    std::ofstream s_flat(flat);
    std::ofstream s_ivf(ivf);
    if (!s_flat || !s_ivf) {
        std::cerr << "failed to open output flat/ivf file." << std::endl;
        return -1;
    }

    size_t flatCnt = 0, ivfCnt = 0;
    std::vector<std::pair<c_t, c_t>> nVec;

    for (const auto& p : cgMap) {
        nVec.push_back(std::make_pair(p.first, p.second.size()));
        if (dry) continue;
        if (p.second.size() > thd) {
            for (const auto& e : p.second) {
                s_ivf << p.first << "|" << e << "|" << gdMap[e] << std::endl;
                ++ivfCnt;
            }
        }
        else {
            for (const auto& e : p.second) {
                s_flat << p.first << "|" << e << "|" << gdMap[e] << std::endl;
                ++flatCnt;
            }
        }
    }

    std::sort(nVec.begin(), nVec.end(), [](std::pair<c_t, c_t> lhs_, std::pair<c_t, c_t> rhs_) {
        return lhs_.second > rhs_.second;
        });

    const size_t topK = 20;
    std::cout << "--------------- cat|count for review ---------" << std::endl;
    for(size_t i = 0; i < topK; ++i) {
        std::cout << nVec[i].first << "|" << nVec[i].second << std::endl;
    }

    if (!dry) std::cout << "flatNum: " << flatCnt << "     ivfNum: " << ivfCnt << std::endl;

    auto diff = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    std::cout << "Elapsed time in seconds: " << diff.count() << std::endl;

    return 0;

}
