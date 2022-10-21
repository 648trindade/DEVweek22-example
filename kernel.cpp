#include "kernel.h"

void move(float3 &position) {
    constexpr float3 velocity{0, -1, 0};
    constexpr float dt{1e-5f};

    position.x = velocity.x * dt;
    position.y = velocity.y * dt;
    position.z = velocity.z * dt;
}

void omp_move(std::vector<float3> &positions) {
#pragma omp parallel for
    for (int i = 0; i < positions.size(); i++) {
        move(positions[i]);
    }
}

void move_positions(std::vector<float3> &positions, bool use_cuda) {
    if (use_cuda) {
        // TODO: add cuda move
    } else {
        omp_move(positions);
    }
}

