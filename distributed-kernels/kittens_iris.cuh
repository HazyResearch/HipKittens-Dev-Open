#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
using namespace kittens;

template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void kittens_store(const GL &dst, const RT &src, const COORD &idx, iris::iris_device_view& iris_ctx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for load/store");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    const int row_offset = src.base_tile_stride*(laneid/src.base_tile_cols);
    const int col_offset = laneid%src.base_tile_cols;
    
    int cur_rank = iris_ctx.cur_rank();

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.base_tile_cols + col_offset;
            #pragma unroll
            for(int k = 0; k < src.base_tile_num_strides; k++) {
                int row = i*src.base_tile_rows + row_offset + k*src.base_tile_elements_per_stride_group;
                #pragma unroll
                for(int l = 0; l < src.base_tile_stride / packing; l++) {
                    int idx = l + k * src.base_tile_stride / packing;
                    
                    // Convert values
                    U val_x = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x);
                    U val_y = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y);
                    
                    // Use iris.store
                    iris_ctx.store(&dst_ptr[(row+l*2)*row_stride + col], val_x, cur_rank);
                    iris_ctx.store(&dst_ptr[(row+l*2+1)*row_stride + col], val_y, cur_rank);
                }
            }
        }
    }
}

// Wrapper without axis template parameter (defaults to axis=2)
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void kittens_store(const GL &dst, const RT &src, const COORD &idx, iris::iris_device_view& iris_ctx) {
    kittens_store<2, RT, GL, COORD>(dst, src, idx, iris_ctx);
}








