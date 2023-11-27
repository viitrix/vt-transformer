#include <time.h>
#include <unistd.h>

#include "vt.hpp"
#include "context.hpp"

namespace vt {

#ifdef _USING_DEVICE_CUDA_
int ComputingContext::cuda_device = -1;
cudaStream_t ComputingContext::cuda_stream = nullptr;
cudaStream_t ComputingContext::assist_streams[ ALL_CUDA_STREAMS ];
cudaEvent_t ComputingContext::events[ ALL_CUDA_EVENTS ];
cublasHandle_t ComputingContext::cublas_handle = nullptr;
cublasLtHandle_t ComputingContext::cublasLt_handle = nullptr;
cudnnHandle_t ComputingContext::cudnn_handle = nullptr;
void* ComputingContext::cuda_workspace = nullptr;
#endif

#ifdef _USING_DEVICE_DCU_
int ComputingContext::dcu_device = -1;
hipStream_t ComputingContext::dcu_stream = nullptr;
#endif

void* ComputingContext::host_workspace = nullptr;
size_t ComputingContext::workspace_size = 0;
std::mt19937* ComputingContext::rng = nullptr;

void ComputingContext::boot(int cud) {
    workspace_size = 1024 * 1024 * 32 * 4;

#ifdef _USING_DEVICE_CUDA_
    cuda_device = cud;

    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    assist_streams[0] = cuda_stream;
    for (int i = 1; i < ALL_CUDA_STREAMS; i++) {
        CUDA_CHECK( cudaStreamCreate(&assist_streams[i]) );
    }
    for (int i = 0; i < ALL_CUDA_EVENTS; i++) {
        CUDA_CHECK( cudaEventCreate(&events[i]) );
    }

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cuda_stream) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));
    CUDA_CHECK( cudaMalloc(&cuda_workspace, workspace_size) );
#endif

#ifdef _USING_DEVICE_DCU_
    dcu_device = cud;

    HIP_CHECK( hipSetDevice(dcu_device) );
    HIP_CHECK( hipStreamCreate(&dcu_stream) );
#endif


    host_workspace = malloc( workspace_size );
    rng = new std::mt19937(1979);
}

void ComputingContext::shutdown() {
    free(host_workspace);

#ifdef _USING_DEVICE_CUDA_
    CUDA_CHECK( cudaFree(cuda_workspace) );
    CUDNN_CHECK( cudnnDestroy(cudnn_handle) );
    CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
    CUBLAS_CHECK( cublasDestroy(cublas_handle) );
    for (int i = 0; i < ALL_CUDA_EVENTS; i++) {
        CUDA_CHECK( cudaEventDestroy(events[i]) );
    }
    CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
    for (int i = 1; i < ALL_CUDA_STREAMS; i++) {
        CUDA_CHECK( cudaStreamDestroy(assist_streams[i]) );
    }
#endif

#ifdef _USING_DEVICE_DCU_
    HIP_CHECK( hipStreamDestroy(dcu_stream) );
#endif
}

#ifdef _USING_DEVICE_CUDA_
float ComputingContext::cuda_event(int flag) {
    if ( flag == 0 ) {
        cudaEventRecord( ComputingContext::events[0] );
        return 0.0;
    }

    if ( flag == 1 ) {
        float delta;
        cudaEventRecord( ComputingContext::events[1] );
        cudaEventSynchronize( ComputingContext::events[0] );
        cudaEventSynchronize( ComputingContext::events[1] );
        cudaEventElapsedTime(&delta,  ComputingContext::events[0],  ComputingContext::events[1]);
        return delta;
    }

    return -1.0;
}
#endif

/**************************************************************/
int      CollectiveContext::current = -1;
#ifdef _USING_HPC_OPENMPI_
int      CollectiveContext::mpi_world = -1;
int      CollectiveContext::mpi_rank = -1;
#endif
int      CollectiveContext::pipe_world = -1;
int      CollectiveContext::pipe_rank = -1;
int*     CollectiveContext::pipe_fds = nullptr;

#ifdef _USING_DEVICE_CUDA_
ncclUniqueId    CollectiveContext::nccl_id;
ncclComm_t      CollectiveContext::nccl_comm = nullptr;
int             CollectiveContext::nccl_rank = -1;
int             CollectiveContext::nccl_world = -1;
#endif

void CollectiveContext::boot_pipe(int gpus) {
    current = time(nullptr);

    pipe_rank = 0;
    pipe_fds = (int *)malloc(sizeof(int) * 2 * (gpus + 1) );
    for (int i = 0; i < gpus + 1; i++) {
        vt_assert( pipe(pipe_fds + i * 2) >= 0, "Can't create pipe between parent and child process!");
    }

#ifdef _USING_DEVICE_CUDA_
    ncclGetUniqueId(&nccl_id);
#endif
    for (int i = 0; i < gpus; i++) {
        int n = fork();
        if ( n == 0 ) {
            pipe_rank = i + 1;
            break;
        }
    }

#ifdef _USING_DEVICE_CUDA_
    // every forked processor has same nccl_id
    if ( pipe_rank >= 1 ) {
        nccl_world = gpus;
        nccl_rank = pipe_rank - 1;

        CUDA_CHECK( cudaSetDevice(nccl_rank) );
        NCCL_CHECK( ncclCommInitRank(&nccl_comm, nccl_world, nccl_id, nccl_rank) );
    } else {
        nccl_comm = nullptr;
    }
#endif

}

#ifdef _USING_HPC_OPENMPI_
void CollectiveContext::boot_mpi(int argc, char* argv[], int gpus) {
    current = time(nullptr);

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    vt_assert( mpi_world == gpus + 1 , "current we only support n + 1 mode!");

#ifdef _USING_DEVICE_CUDA_
    if ( mpi_rank == 0 ) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    if ( mpi_rank >= 1 ) {
        nccl_world = gpus;
        nccl_rank = mpi_rank - 1;

        //ComputingContext::boot(nccl_rank);
        CUDA_CHECK( cudaSetDevice(nccl_rank) );

        NCCL_CHECK(ncclCommInitRank(&nccl_comm, nccl_world, nccl_id, nccl_rank));
    }
#endif

}
#endif

void CollectiveContext::shutdown() {
#ifdef _USING_DEVICE_CUDA_
    if( nccl_comm != nullptr ) {
        NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }
#endif

#ifdef _USING_HPC_OPENMPI_
    if ( mpi_world != -1 ) {
        MPI_Finalize();
    }
#endif

    for (int i = 0; i < pipe_world * 2; i++) {
        close( pipe_fds[i] );
    }
}

int CollectiveContext::pipe_write(const int n, const void *buf, size_t nbyte) {
    if ( pipe_fds == nullptr ) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[n * 2 + 1];
    return write(fd, buf, nbyte);
}

int CollectiveContext::pipe_read(void *buf, size_t nbyte) {
    if ( pipe_fds == nullptr ) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[pipe_rank * 2 + 0];
    return read(fd, buf, nbyte);
}

int CollectiveContext::now() {
    int n = time(nullptr);
    return n - current;
}

/**************************************************************/
const size_t MemoryContext::aligen_size = 4;
size_t MemoryContext::total_size = 0;
size_t MemoryContext::currentp = 0;

void MemoryContext::boot(size_t total_bytes) {
    total_size = total_bytes;
    currentp = 0;
}

void MemoryContext::free(void *m, size_t s) {
    if ( currentp < s) {
        vt_panic("Memory leaking");
    }
    ::free(m);
    currentp -= s;
}

void MemoryContext::shutdown() {

}

void* MemoryContext::alloc(size_t blk_size) {
    vt_assert(blk_size % aligen_size == 0, "block size must be aligend");
    if ( blk_size + currentp > total_size ) {
        vt_panic("Can't allocate memory, out of pre-allocating");
    }

    void* ret = malloc(blk_size);
    return ret;
}

/**************************************************************/
static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
	return as_float(w);
#elif defined(__CUDA_ARCH__)
	return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
	return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return _CopyFloatFromInt32((__int32) w);
#else
	union {
		uint32_t as_bits;
		float as_value;
	} fp32 = { w };
	return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
	return as_uint(f);
#elif defined(__CUDA_ARCH__)
	return (uint32_t) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
	return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return (uint32_t) _CopyInt32FromFloat(f);
#else
	union {
		float as_value;
		uint32_t as_bits;
	} fp32 = { f };
	return fp32.as_bits;
#endif
}

float fp16_to_fp32(local_fp16_t value) {
    uint16_t h = value;
    /*
	 * Extend the half-precision floating-point number to 32 bits and shift to the upper part of the 32-bit word:
	 *      +---+-----+------------+-------------------+
	 *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
	 *      +---+-----+------------+-------------------+
	 * Bits  31  26-30    16-25            0-15
	 *
	 * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
	 */
	const uint32_t w = (uint32_t) h << 16;
	/*
	 * Extract the sign of the input number into the high bit of the 32-bit word:
	 *
	 *      +---+----------------------------------+
	 *      | S |0000000 00000000 00000000 00000000|
	 *      +---+----------------------------------+
	 * Bits  31                 0-31
	 */
	const uint32_t sign = w & UINT32_C(0x80000000);
	/*
	 * Extract mantissa and biased exponent of the input number into the high bits of the 32-bit word:
	 *
	 *      +-----+------------+---------------------+
	 *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
	 *      +-----+------------+---------------------+
	 * Bits  27-31    17-26            0-16
	 */
	const uint32_t two_w = w + w;

	/*
	 * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become mantissa and exponent
	 * of a single-precision floating-point number:
	 *
	 *       S|Exponent |          Mantissa
	 *      +-+---+-----+------------+----------------+
	 *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
	 *      +-+---+-----+------------+----------------+
	 * Bits   | 23-31   |           0-22
	 *
	 * Next, there are some adjustments to the exponent:
	 * - The exponent needs to be corrected by the difference in exponent bias between single-precision and half-precision
	 *   formats (0x7F - 0xF = 0x70)
	 * - Inf and NaN values in the inputs should become Inf and NaN values after conversion to the single-precision number.
	 *   Therefore, if the biased exponent of the half-precision input was 0x1F (max possible value), the biased exponent
	 *   of the single-precision output must be 0xFF (max possible value). We do this correction in two steps:
	 *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset below) rather than by 0x70 suggested
	 *     by the difference in the exponent bias (see above).
	 *   - Then we multiply the single-precision result of exponent adjustment by 2**(-112) to reverse the effect of
	 *     exponent adjustment by 0xE0 less the necessary exponent adjustment by 0x70 due to difference in exponent bias.
	 *     The floating-point multiplication hardware would ensure than Inf and NaN would retain their value on at least
	 *     partially IEEE754-compliant implementations.
	 *
	 * Note that the above operations do not handle denormal inputs (where biased exponent == 0). However, they also do not
	 * operate on denormal inputs, and do not produce denormal results.
	 */
	const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
	const float exp_scale = 0x1.0p-112f;
#else
	const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
	const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	/*
	 * Convert denormalized half-precision inputs into single-precision results (always normalized).
	 * Zero inputs are also handled here.
	 *
	 * In a denormalized number the biased exponent is zero, and mantissa has on-zero bits.
	 * First, we shift mantissa into bits 0-9 of the 32-bit word.
	 *
	 *                  zeros           |  mantissa
	 *      +---------------------------+------------+
	 *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
	 *      +---------------------------+------------+
	 * Bits             10-31                0-9
	 *
	 * Now, remember that denormalized half-precision numbers are represented as:
	 *    FP16 = mantissa * 2**(-24).
	 * The trick is to construct a normalized single-precision number with the same mantissa and thehalf-precision input
	 * and with an exponent which would scale the corresponding mantissa bits to 2**(-24).
	 * A normalized single-precision floating-point number is represented as:
	 *    FP32 = (1 + mantissa * 2**(-23)) * 2**(exponent - 127)
	 * Therefore, when the biased exponent is 126, a unit change in the mantissa of the input denormalized half-precision
	 * number causes a change of the constructud single-precision number by 2**(-24), i.e. the same ammount.
	 *
	 * The last step is to adjust the bias of the constructed single-precision number. When the input half-precision number
	 * is zero, the constructed single-precision number has the value of
	 *    FP32 = 1 * 2**(126 - 127) = 2**(-1) = 0.5
	 * Therefore, we need to subtract 0.5 from the constructed single-precision number to get the numerical equivalent of
	 * the input half-precision number.
	 */
	const uint32_t magic_mask = UINT32_C(126) << 23;
	const float magic_bias = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	/*
	 * - Choose either results of conversion of input as a normalized number, or as a denormalized number, depending on the
	 *   input exponent. The variable two_w contains input exponent in bits 27-31, therefore if its smaller than 2**27, the
	 *   input is either a denormal number, or zero.
	 * - Combine the result of conversion of exponent and mantissa with the sign of the input number.
	 */
	const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
	const uint32_t result = sign |
		(two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

local_fp16_t fp32_to_fp16(float value) {
    float f = value;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
	const float scale_to_inf = 0x1.0p+112f;
	const float scale_to_zero = 0x1.0p-110f;
#else
	const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
	const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
	float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

	const uint32_t w = fp32_to_bits(f);
	const uint32_t shl1_w = w + w;
	const uint32_t sign = w & UINT32_C(0x80000000);
	uint32_t bias = shl1_w & UINT32_C(0xFF000000);
	if (bias < UINT32_C(0x71000000)) {
		bias = UINT32_C(0x71000000);
	}

	base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
	const uint32_t bits = fp32_to_bits(base);
	const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
	const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
	const uint32_t nonsign = exp_bits + mantissa_bits;
	return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}



}
