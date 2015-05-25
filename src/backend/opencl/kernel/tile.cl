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
void tile_kernel(__global T *out,
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
                 const dim_type blocksPerMatX, const dim_type blocksPerMatY)
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

    const dim_type iz = oz % idim2;
    const dim_type iw = ow % idim3;
    const dim_type izw = iw * istride3 + iz * istride2;
    const dim_type ozw = ow * ostride3 + oz * ostride2;

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    for(dim_type oy = yy; oy < odim1; oy += incy) {
        const dim_type iy = oy % idim1;
        for(dim_type ox = xx; ox < odim0; ox += incx) {
            const dim_type ix = ox % idim0;

            dim_type iMem = izw + iy * istride1 + ix;
            dim_type oMem = ozw + oy * ostride1 + ox;

            out[oMem] = in[ioffset + iMem];
        }
    }
}
