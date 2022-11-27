#include "kernel/kernel.h"

#include <cmath>
#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {
    const unsigned int n_positions = std::atoi(argv[1]);
    const bool use_cuda = (argc == 3) && bool(std::atoi(argv[2]));

    srand(0);

    std::vector<float3> positions(n_positions);
    for (unsigned int i = 0; i < n_positions; i++) {
        positions[i] = {
                1000.f * float(rand()) / RAND_MAX,
                1000.f * float(rand()) / RAND_MAX,
                1000.f * float(rand()) / RAND_MAX
        };
    }

    double start = omp_get_wtime();
    move_positions(positions, use_cuda);
    double end = omp_get_wtime();

    std::cout << "Duration: " << (end - start) << " s" << std::endl;
    return 0;
}
