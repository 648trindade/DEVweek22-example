#include "kernel.h"

#include <cstdint>

__host__ __device__ void move(float3 &position) {
    constexpr float3 velocity {0, -1, 0};
    constexpr float dt {1e-5f};

    position.x = velocity.x * dt;
    position.y = velocity.y * dt;
    position.z = velocity.z * dt;
}

__global__ void dispatch_move_kernel(float3 *positions, uint32_t size) {
    const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t step = blockDim.x;

    for (uint32_t i = start; i < size; i += step) {
        move(positions[i]);
    }
}

void cuda_move(std::vector<float3> &positions) {
    float3 *device_positions;
    cudaMalloc(&device_positions, positions.size() * sizeof(float3));
    cudaMemcpy(device_positions, positions.data(),
               positions.size() * sizeof(float3), cudaMemcpyHostToDevice);

    dispatch_move_kernel<<<1, 1>>>(device_positions, positions.size());

    cudaMemcpy(positions.data(), device_positions,
               positions.size() * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(device_positions);
}

void omp_move(std::vector<float3> &positions) {
    #pragma omp parallel for
    for (int i = 0; i < positions.size(); i++) {
        move(positions[i]);
    }
}

void move_positions(std::vector<float3> &positions, bool use_cuda) {
    if (use_cuda) {
        cuda_move(positions);
    } else {
        omp_move(positions);
    }
}
