namespace vt { namespace cuda {

const int Q4_BLOCK_SIZE = 128;
typedef struct {
    float   d;
    float   m;
    uint8_t q[64];
} q4_block_t;
static_assert(sizeof(q4_block_t) == sizeof(float) * 2 + 64, "wrong q4_block_t size/padding");

typedef struct {
    float   d;
    float   m;
    uint8_t q[0];
} q8_head_t;


}}
