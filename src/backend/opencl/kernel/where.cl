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

#if CPLX
#define isZero(val) ((val.x ==0) && (val.y == 0))
#else
#define isZero(val) ((val == 0))
#endif

__kernel
void get_out_idx_kernel(__global float *oData,
                        __global float *otData,
                        const dim_type otdim0,
                        const dim_type otdim1,
                        const dim_type otdim2,
                        const dim_type otdim3,
                        const dim_type otstride0,
                        const dim_type otstride1,
                        const dim_type otstride2,
                        const dim_type otstride3,
                        __global float *rtData,
                        const dim_type rtstride0,
                        const dim_type rtstride1,
                        const dim_type rtstride2,
                        const dim_type rtstride3,
                        __global T *iData,
                        const dim_type istride0,
                        const dim_type istride1,
                        const dim_type istride2,
                        const dim_type istride3,
                        uint groups_x,
                        uint groups_y,
                        uint lim)
{
    const uint lidx = get_local_id(0);
    const uint lidy = get_local_id(1);

    const uint zid = get_group_id(0) / groups_x;
    const uint wid = get_group_id(1) / groups_y;
    const uint groupId_x = get_group_id(0) - (groups_x) * zid;
    const uint groupId_y = get_group_id(1) - (groups_y) * wid;
    const uint xid = groupId_x * get_local_size(0) * lim + lidx;
    const uint yid = groupId_y * get_local_size(1) + lidy;

    const uint off = wid * otstride3 + zid * otstride2 + yid * otstride1;
    const uint gid = wid * rtstride3 + zid * rtstride2 + yid * rtstride1 + groupId_x;

    otData += wid * otstride3 + zid * otstride2 + yid * otstride1;
    iData  += wid *  istride3 + zid *  istride2 + yid *  istride1;

    bool cond = (yid < otdim1) && (zid < otdim2) && (wid < otdim3);
    if (!cond) return;

    uint accum = (gid == 0) ? 0 : rtData[gid - 1];

    for (uint k = 0, id = xid;
         k < lim && id < otdim0;
         k++, id += get_local_size(0)) {

        uint idx = otData[id] + accum;
        T ival = iData[id];
        if (!isZero(ival)) oData[idx - 1] = (off + id);
    }
}
