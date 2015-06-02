/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef To
#define To float
#endif

#ifndef Ti
#define Ti float
#endif

#ifndef T
#define T float
#endif

#ifndef dim_type
#define dim_type int
#endif

#ifndef THREADS_PER_GROUP
#define THREADS_PER_GROUP 256
#endif

#ifndef DIMX
#define DIMX 32
#endif

#include "iops.cl"

#define INSTANTIATE(OP) \
__kernel __attribute__ ((reqd_work_group_size(32, 8, 1))) \
void ireduce_first_kernel_##OP(__global T *oData, \
                               const dim_type ostride0, \
                               const dim_type ostride1, \
                               const dim_type ostride2, \
                               const dim_type ostride3, \
                               const dim_type ooffset, \
                               __global T *olData, \
                               const __global T *iData, \
                               const dim_type idim0, \
                               const dim_type idim1, \
                               const dim_type idim2, \
                               const dim_type idim3, \
                               const dim_type istride0, \
                               const dim_type istride1, \
                               const dim_type istride2, \
                               const dim_type istride3, \
                               const dim_type ioffset, \
                               const __global T *ilData, \
                               uint groups_x, uint groups_y, uint repeat, \
                               T init, int IS_FIRST) \
{ \
    const uint lidx = get_local_id(0); \
    const uint lidy = get_local_id(1); \
    const uint lid  = lidy * get_local_size(0) + lidx; \
 \
    const uint zid = get_group_id(0) / groups_x; \
    const uint wid = get_group_id(1) / groups_y; \
    const uint groupId_x = get_group_id(0) - (groups_x) * zid; \
    const uint groupId_y = get_group_id(1) - (groups_y) * wid; \
    const uint xid = groupId_x * get_local_size(0) * repeat + lidx; \
    const uint yid = groupId_y * get_local_size(1) + lidy; \
 \
    iData += wid * istride3 + zid * istride2 + \
        yid * istride1 + ioffset; \
 \
    if (!IS_FIRST) { \
        ilData += wid * istride3 + zid * istride2 + \
            yid * istride1 + ioffset; \
    } \
 \
    oData += wid * ostride3 + zid * ostride2 + \
        yid * ostride1 + ooffset; \
 \
    olData += wid * ostride3 + zid * ostride2 + \
        yid * ostride1 + ooffset; \
 \
    bool cond  = (yid < idim1); \
         cond &= (zid < idim2); \
         cond &= (wid < idim3); \
 \
    __local T s_val[THREADS_PER_GROUP]; \
    __local T s_idx[THREADS_PER_GROUP]; \
 \
    int last = (xid + repeat * DIMX); \
    int lim = last > idim0 ? idim0 : last; \
    T out_val = init; \
    T out_idx = xid; \
 \
    if (cond && xid < lim) { \
        out_val = iData[xid]; \
        if (!IS_FIRST) out_idx = ilData[xid]; \
    } \
 \
    for (int id = xid + DIMX; cond && id < lim; id += DIMX) { \
        if (IS_FIRST != 0) \
            bin_##OP(&out_val, &out_idx, iData[id], id); \
        else \
            bin_##OP(&out_val, &out_idx, iData[id], ilData[id]); \
    } \
 \
    s_val[lid] = out_val; \
    s_idx[lid] = out_idx; \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    __local T *s_vptr = s_val + lidy * DIMX; \
    __local T *s_iptr = s_idx + lidy * DIMX; \
 \
    if (DIMX == 256) { \
        if (lidx < 128) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[lidx + 128], s_iptr[lidx + 128]); \
            s_vptr[lidx] = out_val; \
            s_iptr[lidx] = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMX >= 128) { \
        if (lidx <  64) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[lidx +  64], s_iptr[lidx +  64]); \
            s_vptr[lidx] = out_val; \
            s_iptr[lidx] = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMX >=  64) { \
        if (lidx <  32) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[lidx +  32], s_iptr[lidx +  32]); \
            s_vptr[lidx] = out_val; \
            s_iptr[lidx] = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (lidx <  16) { \
        bin_##OP(&out_val, &out_idx, \
              s_vptr[lidx +  16], s_iptr[lidx +  16]); \
        s_vptr[lidx] = out_val; \
        s_iptr[lidx] = out_idx; \
    } \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (lidx <   8) { \
        bin_##OP(&out_val, &out_idx, \
              s_vptr[lidx +   8], s_iptr[lidx +   8]); \
        s_vptr[lidx] = out_val; \
        s_iptr[lidx] = out_idx; \
    } \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (lidx <   4) { \
        bin_##OP(&out_val, &out_idx, \
              s_vptr[lidx +   4], s_iptr[lidx +   4]); \
        s_vptr[lidx] = out_val; \
        s_iptr[lidx] = out_idx; \
    } \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (lidx <   2) { \
        bin_##OP(&out_val, &out_idx, \
              s_vptr[lidx +   2], s_iptr[lidx +   2]); \
        s_vptr[lidx] = out_val; \
        s_iptr[lidx] = out_idx; \
    } \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (lidx <   1) { \
        bin_##OP(&out_val, &out_idx, \
              s_vptr[lidx +   1], s_iptr[lidx +   1]); \
        s_vptr[lidx] = out_val; \
        s_iptr[lidx] = out_idx; \
    } \
 \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (cond && lidx == 0) { \
        oData[groupId_x] = s_vptr[0]; \
        olData[groupId_x] = s_iptr[0]; \
    } \
}

INSTANTIATE(MIN_OP)
INSTANTIATE(MAX_OP)
