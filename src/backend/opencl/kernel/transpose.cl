/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef T
#define T float
#endif

#ifndef dim_type
#define dim_type int
#endif

#define TILE_DIM 32
#define THREADS_X TILE_DIM
#define THREADS_Y (TILE_DIM / 4)

__kernel __attribute__ ((reqd_work_group_size(32, 8, 1)))
void transpose(__global T *oData,
               const dim_type odim0,
               const dim_type odim1,
               const dim_type odim2,
               const dim_type odim3,
               const dim_type ostride0,
               const dim_type ostride1,
               const dim_type ostride2,
               const dim_type ostride3,
               const dim_type ooffset,
               const __global T *iData,
               const dim_type idim0,
               const dim_type idim1,
               const dim_type idim2,
               const dim_type idim3,
               const dim_type istride0,
               const dim_type istride1,
               const dim_type istride2,
               const dim_type istride3,
               const dim_type ioffset,
               const dim_type blocksPerMatX,
               const dim_type blocksPerMatY,
               const unsigned is32Multiple)
{
    __local T shrdMem[TILE_DIM*(TILE_DIM+1)];

    const dim_type shrdStride = TILE_DIM+1;
    // create variables to hold output dimensions
    const dim_type oDim0 = odim0;
    const dim_type oDim1 = odim1;
    const dim_type iDim0 = idim0;
    const dim_type iDim1 = idim1;

    // calculate strides
    const dim_type oStride1 = ostride1;
    const dim_type iStride1 = istride1;

    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    // batch based block Id
    const dim_type batchId_x  = get_group_id(0) / blocksPerMatX;
    const dim_type blockIdx_x = (get_group_id(0) - batchId_x * blocksPerMatX);

    const dim_type batchId_y  = get_group_id(1) / blocksPerMatY;
    const dim_type blockIdx_y = (get_group_id(1) - batchId_y * blocksPerMatY);

    const dim_type x0 = TILE_DIM * blockIdx_x;
    const dim_type y0 = TILE_DIM * blockIdx_y;

    // calculate global indices
    dim_type gx = lx + x0;
    dim_type gy = ly + y0;

    // offset in and out based on batch id
    // also add the subBuffer offsets
    iData += batchId_x *  istride2 + batchId_y *  istride3 +  ioffset;
    oData += batchId_x * ostride2 + batchId_y * ostride3 + ooffset;

    for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        dim_type gy_ = gy + repeat;
        if (is32Multiple != 0 || (gx < iDim0 && gy_ < iDim1))
            shrdMem[(ly + repeat) * shrdStride + lx] = iData[gy_ * iStride1 + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    gx = lx + y0;
    gy = ly + x0;

    for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        dim_type gy_ = gy + repeat;
        if (is32Multiple != 0 || (gx < oDim0 && gy_ < oDim1)) {
            oData[gy_ * oStride1 + gx] = doOp(shrdMem[lx * shrdStride + ly + repeat]);
        }
    }
}
