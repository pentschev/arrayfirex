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

__kernel __attribute__ ((reqd_work_group_size(32, 8, 1)))
void reorder_kernel(__global T *out,
                    const dim_type odim0,
                    const dim_type odim1,
                    const dim_type odim2,
                    const dim_type odim3,
                    const dim_type ostride0,
                    const dim_type ostride1,
                    const dim_type ostride2,
                    const dim_type ostride3,
                    const dim_type ooffset,
                    __global const T *in,
                    const dim_type idim0,
                    const dim_type idim1,
                    const dim_type idim2,
                    const dim_type idim3,
                    const dim_type istride0,
                    const dim_type istride1,
                    const dim_type istride2,
                    const dim_type istride3,
                    const dim_type ioffset,
                    const dim_type d0,
                    const dim_type d1,
                    const dim_type d2,
                    const dim_type d3,
                    const dim_type blocksPerMatX,
                    const dim_type blocksPerMatY)
{
    const dim_type oz = get_group_id(0) / blocksPerMatX;
    const dim_type ow = get_group_id(1) / blocksPerMatY;

    const dim_type blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const dim_type blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const dim_type xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(xx >= odim0 ||
       yy >= odim1 ||
       oz >= odim2 ||
       ow >= odim3)
        return;

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    const dim_type o_off   = ow * ostride3 + oz * ostride2;
    const dim_type rdims[] = {d0, d1, d2, d3};
          dim_type ods[]   = {xx, yy, oz, ow};
          dim_type ids[4]  = {0};

    ids[rdims[3]] = ow;
    ids[rdims[2]] = oz;

    for(dim_type oy = yy; oy < odim1; oy += incy) {
        ids[rdims[1]] = oy;
        for(dim_type ox = xx; ox < odim0; ox += incx) {
            ids[rdims[0]] = ox;

            const dim_type oIdx = o_off + oy * ostride1 + ox;

            const dim_type iIdx = ids[3] * istride3 + ids[2] * istride2 +
                                  ids[1] * istride1 + ids[0];

            out[oIdx] = in[ioffset + iIdx];
        }
    }
}
