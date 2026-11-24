// Exact small top-k candidates for the 4.0.5 TP4 sampled head.
// Comparisons only: no approximate softmax, exponential or RNG on the GPU.
// Top-k order is (descending float value, ascending global token ID), with
// signed zero tied. Original float bits are retrieved for the host sampler.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

struct Pair { float value; uint32_t id; };
static constexpr int K = 64;
static constexpr int TILE = 1024;
static constexpr int MAX_TILES = 32;
static constexpr int PACKET = K + 1;
static_assert(sizeof(Pair) == 8, "host ABI");

__device__ __forceinline__ uint64_t order_key(float value, uint32_t id) {
    uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    uint32_t ordered = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return (uint64_t(ordered) << 32) | uint64_t(~id);
}

template<int N>
__device__ __forceinline__ void descending(uint64_t* values) {
    // Each compare pair has one writer. Barriers separate sorting stages.
    #pragma unroll
    for (unsigned size = 2; size <= N; size <<= 1) {
        #pragma unroll
        for (unsigned distance = size >> 1; distance; distance >>= 1) {
            for (unsigned i = threadIdx.x; i < N; i += blockDim.x) {
                unsigned j = i ^ distance;
                if (j > i) {
                    uint64_t a = values[i], b = values[j];
                    bool down = (i & size) == 0;
                    if (down ? a < b : a > b) { values[i] = b; values[j] = a; }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void tf_405_sample_tiles(const float* logits,
        uint64_t stride, uint32_t columns, uint32_t offset, uint32_t logical_vocab,
        Pair* scratch, uint32_t* invalid) {
    __shared__ uint64_t keys[TILE];
    const unsigned row = blockIdx.y, tile = blockIdx.x;
    const unsigned tiles = (columns + TILE - 1) / TILE;
    for (unsigned i = threadIdx.x; i < TILE; i += blockDim.x) {
        const unsigned local = tile * TILE + i, id = offset + local;
        uint64_t key = 0;
        if (local < columns && id < logical_vocab) {
            const float v = logits[uint64_t(row) * stride + local];
            const uint32_t bits = __float_as_uint(v);
            const bool bad = ((bits & 0x7fffffffu) > 0x7f800000u) || bits == 0x7f800000u;
            if (bad) atomicOr(invalid + row, 1u);
            else key = order_key(v, id);
        }
        keys[i] = key;
    }
    __syncthreads();
    descending<TILE>(keys);
    if (threadIdx.x < K) {
        const uint64_t key = keys[threadIdx.x];
        const uint32_t id = ~uint32_t(key);
        scratch[(uint64_t(row) * tiles + tile) * K + threadIdx.x] = key == 0
            ? Pair{-CUDART_INF_F, 0xffffffffu}
            : Pair{logits[uint64_t(row) * stride + id - offset], id};
    }
}

extern "C" __global__ void tf_405_sample_merge(const float* logits,
        uint64_t stride, uint32_t columns, uint32_t offset,
        const Pair* scratch, const uint32_t* invalid, Pair* result) {
    __shared__ uint64_t keys[MAX_TILES * K];
    const unsigned row = blockIdx.x;
    const unsigned tiles = (columns + TILE - 1) / TILE;
    for (unsigned i = threadIdx.x; i < MAX_TILES * K; i += blockDim.x) {
        uint64_t key = 0;
        if (i < tiles * K) {
            const Pair pair = scratch[uint64_t(row) * tiles * K + i];
            if (pair.id != 0xffffffffu) key = order_key(pair.value, pair.id);
        }
        keys[i] = key;
    }
    __syncthreads();
    descending<MAX_TILES * K>(keys);
    if (threadIdx.x < K) {
        const uint64_t key = keys[threadIdx.x];
        const uint32_t id = ~uint32_t(key);
        result[uint64_t(row) * PACKET + threadIdx.x] = key == 0
            ? Pair{-CUDART_INF_F, 0xffffffffu}
            : Pair{logits[uint64_t(row) * stride + id - offset], id};
    }
    if (threadIdx.x == K) result[uint64_t(row) * PACKET + K] = Pair{0.0f, invalid[row]};
}
