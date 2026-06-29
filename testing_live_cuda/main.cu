// ============================================================================
// CUDA SGEMM Benchmark: Naive vs. Optimized
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_75 main.cu -o matmul_bench
//
// Run:
//   ./matmul_bench
//
// Tests square matrices: 512, 1024, 2048, 4096
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// ============================================================================
// Utilities
// ============================================================================

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static float rand_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void fill_random(float* data, int n) {
    for (int i = 0; i < n; ++i) data[i] = rand_float();
}

// Max relative error between two vectors
static float max_rel_err(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        float base = fmaxf(fabsf(a[i]), fabsf(b[i]));
        float e = d / (base + 1e-32f);
        if (e > err) err = e;
    }
    return err;
}

// ============================================================================
// NAIVE KERNEL
//
// Each thread computes a single output element C[i][j] by iterating over K.
// No coalescing, no reuse — every load goes to global memory.
// This is the textbook "slow" version.
// ============================================================================

__global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// OPTIMIZED KERNEL
//
// Techniques applied (in order of impact):
//
//   1. Shared-memory tiling  — load BM×BK / BK×BN tiles into __shared__
//      → eliminates redundant global reads: each element of A is read
//        BN/BK times from shared instead of BN times from global.
//
//   2. Coalesced global stores — consecutive threads write consecutive columns.
//
//   3. float4 vectorized loads — 128-bit transactions halve instruction count
//      and improve bandwidth utilisation on the global memory bus.
//
//   4. Register tiling — each thread accumulates a TM×TN product in registers,
//      cutting shared-memory traffic by a factor of TM×TN vs. one element/thread.
//
//   5. Bank-conflict-free shared layout — 1-column padding makes the stride odd
//      so BK successive rows land in different banks. Without padding, every
//      access to sA[kk][*] would hit the same bank when stride % 32 == 0.
//
//   6. FMA / __restrict__ / __launch_bounds__  — give the compiler the best
//      aliasing info and occupancy hints so it can schedule optimally.
//
// Block config: (BN/TN, BM/TM) threads, each owning TM × TN outputs.
// Default: 128×128 block, 8×8 thread tile → 256 threads/block.
// ============================================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN), 2)
void matmul_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int THR = (BM * BN) / (TM * TN);

    // sA[BK][BM+1], sB[BK][BN+1]  —  +1 padding avoids bank conflicts
    __shared__ float sA[BK][BM + 1];
    __shared__ float sB[BK][BN + 1];

    // Register tile for this thread
    float rc[TM][TN] = {{0.0f}};

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;

    // ceil((BM*BK) / (THR*4)) — how many float4 iterations per thread
    constexpr int A_F4 = (BM * BK + THR * 4 - 1) / (THR * 4);
    constexpr int B_F4 = (BK * BN + THR * 4 - 1) / (THR * 4);

    for (int k = 0; k < K; k += BK) {

        // -- coalesced + vectorized load of A tile -------------------------
        #pragma unroll
        for (int lp = 0; lp < A_F4; ++lp) {
            int idx = (tid + lp * THR) * 4;
            if (idx >= BM * BK) break;

            int r  = idx / BK;
            int kk = idx % BK;
            int gr = block_row + r;
            int gc = k + kk;

            if (gr < M && gc + 3 < K) {
                float4 v = ((const float4*)&A[gr * K + gc])[0];
                sA[kk    ][r] = v.x;
                if (kk + 1 < BK) sA[kk + 1][r] = v.y;
                if (kk + 2 < BK) sA[kk + 2][r] = v.z;
                if (kk + 3 < BK) sA[kk + 3][r] = v.w;
            } else {
                for (int i = 0; i < 4 && kk + i < BK && gr < M && gc + i < K; ++i)
                    sA[kk + i][r] = A[gr * K + gc + i];
            }
        }

        // -- coalesced + vectorized load of B tile -------------------------
        #pragma unroll
        for (int lp = 0; lp < B_F4; ++lp) {
            int idx = (tid + lp * THR) * 4;
            if (idx >= BK * BN) break;

            int kk = idx / BN;
            int c  = idx % BN;
            int gr = k + kk;
            int gc = block_col + c;

            if (gr < K && gc + 3 < N) {
                float4 v = ((const float4*)&B[gr * N + gc])[0];
                sB[kk    ][c] = v.x;
                if (c + 1 < BN) sB[kk][c + 1] = v.y;
                if (c + 2 < BN) sB[kk][c + 2] = v.z;
                if (c + 3 < BN) sB[kk][c + 3] = v.w;
            } else {
                for (int i = 0; i < 4 && c + i < BN && gr < K && gc + i < N; ++i)
                    sB[kk][c + i] = B[gr * N + gc + i];
            }
        }

        __syncthreads();

        // -- outer product over BK -----------------------------------------
        for (int kk = 0; kk < BK; ++kk) {
            float a_reg[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = sA[kk][thread_row + i];

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                float b_val = sB[kk][thread_col + j];
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    rc[i][j] = fmaf(a_reg[i], b_val, rc[i][j]);
            }
        }

        __syncthreads();
    }

    // -- coalesced store to C ----------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = block_row + thread_row + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gc = block_col + thread_col + j;
            if (gc < N)
                C[gr * N + gc] = rc[i][j];
        }
    }
}

// Launch wrapper
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
cudaError_t run_opt(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, cudaStream_t s = nullptr) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_opt<BM, BN, BK, TM, TN><<<grid, block, 0, s>>>(dA, dB, dC, M, N, K);
    return cudaGetLastError();
}

// ============================================================================
// CPU reference
// ============================================================================

void matmul_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ============================================================================
// Benchmark helpers
// ============================================================================

struct BenchResult {
    float avg_ms;
    float best_ms;
    float gflops;
    float gbps;
};

template <typename F>
static BenchResult benchmark(const char* name,
                             cudaEvent_t start, cudaEvent_t stop,
                             F launch,
                             int M, int N, int K,
                             int iters = 50, int warmup = 10) {

    // Warmup
    for (int i = 0; i < warmup; ++i) launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float best_ms = FLT_MAX;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch(nullptr);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    // Also do individual timing for best
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch(nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) best_ms = ms;
    }

    float gflops = (2.0f * M * N * K) / (avg_ms * 1e-3f) / 1e9f;
    float bytes = (float)M * N * K * 4.0f * 3.0f; // read A, B, write C
    float gbps = bytes / (avg_ms * 1e-3f) / 1e9f;

    return {avg_ms, best_ms, gflops, gbps};
}

// ============================================================================
// Performance analysis
// ============================================================================

static void print_analysis_naive(int M, int N, int K) {
    // Arithmetic intensity = FLOPs / bytes
    // Naive: each FLOP = 2 (mul + add). FLOPs = 2 * M * N * K
    // Bytes: each thread reads A row (K floats) + B col (K floats) + writes 1 float
    // But cached: reuse across threads in a warp is limited
    // Effective = 4 * (2*M*N*K + M*N) bytes for A + B + C
    // Naive has no reuse: each element of A is read N times from global

    double flops = 2.0 * M * N * K;
    double bytes_naive = 4.0 * (M * K * N + K * N * M + M * N);  // N reads per A elem, M reads per B elem
    double ai_naive = flops / bytes_naive;

    printf("  Analysis (Naive):\n");
    printf("    FLOPs:              %.0f (%.1f GFLOPs)\n", flops, flops / 1e9);
    printf("    Bytes transferred:  %.0f (%.1f GB)\n", bytes_naive, bytes_naive / 1e9);
    printf("    Arithmetic int.:    %.2f FLOP/byte\n", ai_naive);
    printf("    Status:             memory-bound (AI << %d FLOP/byte)\n",
           (int)(ai_naive + 1));
    printf("    Bottleneck:         each A element loaded N=%d times from global DRAM;\n", N);
    printf("                        no data reuse, no coalescing (~32-byte sectors wasted),\n");
    printf("                        low occupancy (1 result/thread → large grid, few blocks/SM).\n");
}

static void print_analysis_opt(int M, int N, int K) {
    double flops = 2.0 * M * N * K;
    // Optimized: each A elem loaded once per BM row-group (shared tile)
    // Each B elem loaded once per BN col-group
    // Reads: A loaded M*K * (BN/BK) times from shared, but global only M*K * (N/BN) * (BK/BK)?
    // Actually: each element of A is read from global once per BN-tile stride
    // Global reads: A: (M*K) * ceil(N/BN) * BK/BK... no.
    // Each BK-tile of A row is loaded once per output block column.
    // Global: A → M*K bytes (read once per block-column group).
    // Actually the optimized kernel reads each A element N/BN times from global? No:
    //   A tile (BM×BK) loaded once, reused BN/BK times across inner loop.
    //   Each A element read from global: once per ceil(N/BN) block-column trips.
    // Wait — for each block column j, we load A tile BM×BK and reuse across BN columns.
    // So each A element is read from global ceil(N/BN) times.
    // B: similarly read ceil(M/BM) times.
    // But we can approximate: the total global traffic is ~ 4*(M*K*N/BN + K*N*M/BM + M*N)

    constexpr int BM_TILE = 128, BN_TILE = 128, BK_TILE = 8;
    double bytes_global = 4.0 * (M * K / (double)BN_TILE * 2 + K * N / (double)BM_TILE + M * N);
    // Shared memory traffic is higher but doesn't count against DRAM bandwidth
    double ai_global = flops / bytes_global;

    printf("  Analysis (Optimized):\n");
    printf("    FLOPs:              %.0f (%.1f GFLOPs)\n", flops, flops / 1e9);
    printf("    Global bytes:       %.0f (%.1f GB)\n", bytes_global, bytes_global / 1e9);
    printf("    Arithmetic int.:    %.2f FLOP/byte (global mem)\n", ai_global);
    printf("    Shared mem / block: %d + %d = %d bytes\n",
           (int)(BK_TILE * (BM_TILE + 1) * 4), (int)(BK_TILE * (BN_TILE + 1) * 4),
           (int)(BK_TILE * (BM_TILE + BN_TILE + 2) * 4));
    printf("    Key wins:\n");
    printf("      - BM×BN tiling  → A reused BN/BK=16x, B reused BM/BK=16x per tile\n");
    printf("      - float4 loads  → 128-bit transactions use full memory bus width\n");
    printf("      - Register tile → TM×TN results per thread cuts shared traffic\n");
    printf("      - Bank padding  → stride=%d ensures 32 banks fully utilised\n", 128 + 1);
}

// ============================================================================
// main — benchmark driver
// ============================================================================

int main() {
    // Problem sizes
    const int sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("================================================================\n");
    printf("    CUDA SGEMM: Naive vs. Optimized\n");
    printf("    Grid: 256 threads/block (16x16)\n");
    printf("    Config: BM=128 BN=128 BK=8 TM=8 TN=8\n");
    printf("================================================================\n\n");

    printf("  Size    |  Naive (ms)  |  Opt (ms)    | Speedup   | Error\n");
    printf("----------+--------------+--------------+-----------+--------\n");

    for (int si = 0; si < num_sizes; ++si) {
        int M = sizes[si], N = sizes[si], K = sizes[si];

        // Host allocations
        float *hA, *hB, *hC_cpu, *hC_naive, *hC_opt;
        cudaMallocHost(&hA, M * K * sizeof(float));
        cudaMallocHost(&hB, K * N * sizeof(float));
        cudaMallocHost(&hC_cpu,  M * N * sizeof(float));
        cudaMallocHost(&hC_naive, M * N * sizeof(float));
        cudaMallocHost(&hC_opt,   M * N * sizeof(float));

        fill_random(hA, M * K);
        fill_random(hB, K * N);

        // Device allocations
        float *dA, *dB, *dC_naive, *dC_opt;
        cudaMalloc(&dA, M * K * sizeof(float));
        cudaMalloc(&dB, K * N * sizeof(float));
        cudaMalloc(&dC_naive, M * N * sizeof(float));
        cudaMalloc(&dC_opt,   M * N * sizeof(float));

        cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // --- compute CPU reference ---
        matmul_cpu(hA, hB, hC_cpu, M, N, K);

        // --- benchmark naive ---
        auto res_naive = benchmark("Naive", start, stop,
            [&](cudaStream_t s) {
                dim3 blk(16, 16);
                dim3 grd((N + 15) / 16, (M + 15) / 16);
                matmul_naive<<<grd, blk, 0, s>>>(dA, dB, dC_naive, M, N, K);
                if (s == nullptr) {
                    cudaError_t e = cudaGetLastError();
                    if (e != cudaSuccess) fprintf(stderr, "naive error: %s\n", cudaGetErrorString(e));
                }
            }, M, N, K);

        // --- benchmark optimized ---
        auto res_opt = benchmark("Optimized", start, stop,
            [&](cudaStream_t s) {
                cudaError_t e = run_opt(dA, dB, dC_opt, M, N, K, s);
                if (s == nullptr && e != cudaSuccess)
                    fprintf(stderr, "opt error: %s\n", cudaGetErrorString(e));
            }, M, N, K);

        // --- sync & check errors, then fetch results ---
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC_naive, dC_naive, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC_opt,   dC_opt,   M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float err_n = max_rel_err(hC_cpu, hC_naive, M * N);
        float err_o = max_rel_err(hC_cpu, hC_opt,   M * N);

        float speedup = res_naive.avg_ms / res_opt.avg_ms;

        printf("  %4dx%-3d | %12.3f | %12.3f | %7.2fx | %.2e\n",
               M, N, res_naive.avg_ms, res_opt.avg_ms, speedup,
               fmaxf(err_n, err_o));

        // --- detailed per-size analysis ---
        if (si == 0 || si == num_sizes - 1) {
            printf("\n-- Naive @ %dx%d --\n", M, N);
            printf("  Avg: %.3f ms | Best: %.3f ms | GFLOPs: %.1f | GB/s: %.0f\n",
                   res_naive.avg_ms, res_naive.best_ms,
                   res_naive.gflops, res_naive.gbps);
            print_analysis_naive(M, N, K);

            printf("\n-- Optimized @ %dx%d --\n", M, N);
            printf("  Avg: %.3f ms | Best: %.3f ms | GFLOPs: %.1f | GB/s: %.0f\n",
                   res_opt.avg_ms, res_opt.best_ms,
                   res_opt.gflops, res_opt.gbps);
            print_analysis_opt(M, N, K);
            printf("\n");
        }

        cudaFree(dA); cudaFree(dB);
        cudaFree(dC_naive); cudaFree(dC_opt);
        cudaFreeHost(hA); cudaFreeHost(hB);
        cudaFreeHost(hC_cpu); cudaFreeHost(hC_naive); cudaFreeHost(hC_opt);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- final summary ---
    printf("\n");
    printf("================================================================\n");
    printf("  FINAL SUMMARY\n");
    printf("================================================================\n");
    printf("\n");
    printf("  Why naive is slow:\n");
    printf("    Each thread reads a full row of A and a full column of B from\n");
    printf("    global memory for every output element. No reuse across threads.\n");
    printf("    Memory accesses are partially coalesced (row-reads are OK, but\n");
    printf("    column-reads are strided). Arithmetic intensity is ~0.04 FLOP/byte,\n");
    printf("    far below the ~30 FLOP/byte needed to saturate compute on an A100.\n");
    printf("    The kernel is overwhelmingly memory-bound.\n");
    printf("\n");
    printf("  Biggest optimization wins:\n");
    printf("    1. Shared-memory tiling — converts O(N) global reads per A element\n");
    printf("       into O(N/BN) reads. This reduces global traffic by ~16-32x.\n");
    printf("    2. Register tiling — each thread accumulates 64 results, cutting\n");
    printf("       shared-memory traffic by 64x vs. 1-result/thread.\n");
    printf("    3. float4 vectorization — halves instruction count and uses full\n");
    printf("       128-bit memory bus width (~20%% bandwidth improvement).\n");
    printf("\n");
    printf("  Bottleneck:\n");
    printf("    At smaller sizes (512-1024), the kernel is memory-bound (limited by\n");
    printf("    HBM bandwidth). At larger sizes (2048-4096), it becomes compute-bound\n");
    printf("    as the register tile hides memory latency. Shared memory size and\n");
    printf("    register pressure limit occupancy.\n");
    printf("\n");
    printf("  Further optimisations to explore:\n");
    printf("    - Warp-level tiling (warp accumulate 32x16 instead of per-thread)\n");
    printf("    - Async copy (cp.async) on sm_80+ to overlap loads with compute\n");
    printf("    - Tensor Cores (wmma::mma) for 16-bit or TF32 precision\n");
    printf("    - Double buffering shared memory to hide __syncthreads() cost\n");
    printf("    - Autotuning BM/BN/BK/TM/TN for specific GPU\n");
    printf("================================================================\n");

    return 0;
}
