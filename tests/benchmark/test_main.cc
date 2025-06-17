#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "alog/Logger.h"
#include "alog/Configurator.h"

int main(int argc, char* argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    testing::InitGoogleTest(&argc, argv);

    alog::Configurator::configureLogger("tests/benchmark/log.conf");

    int ret = RUN_ALL_TESTS();

    alog::Logger::shutdown();
    return ret;
}
