#include <iostream>
#include <vector>
#include "fem.h"

int main(int argc, char *argv[]) {
    

    Timer timer;
    timer.start();
    std::string file(argv[1]);
    FEM fem ;
    fem.initFEM(file);
    fem.solve();
    std::cout << "Duration: " << timer.stop() << std::endl;

    return 0;
}
