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

#include "ops.cl"

#define INSTANTIATE(OP) \
__kernel __attribute__ ((reqd_work_group_size(32, 8, 1))) \
void reduce_first_kernel_##OP(__global To *oData, \
                                const dim_type odim0, \
                                const dim_type odim1, \
                                const dim_type odim2, \
                                const dim_type odim3, \
                                const dim_type ostride0, \
                                const dim_type ostride1, \
                                const dim_type ostride2, \
                                const dim_type ostride3, \
                                const dim_type ooffset, \
                                const __global Ti *iData, \
                                const dim_type idim0, \
                                const dim_type idim1, \
                                const dim_type idim2, \
                                const dim_type idim3, \
                                const dim_type istride0, \
                                const dim_type istride1, \
                                const dim_type istride2, \
                                const dim_type istride3, \
                                const dim_type ioffset, \
                                uint groups_x, uint groups_y, uint repeat, \
                                const To init) \
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
    oData += wid * ostride3 + zid * ostride2 + \
        yid * ostride1 + ooffset; \
\
    int cond = (yid < idim1); \
    cond    &= (zid < idim2); \
    cond    &= (wid < idim3); \
\
    __local To s_val[THREADS_PER_GROUP]; \
\
    int last = (xid + repeat * DIMX); \
    int lim = last > idim0 ? idim0 : last; \
    To out_val = init; \
\
    for (int id = xid; cond != 0 && id < lim; id += DIMX) { \
        To in_val = transform_##OP(iData[id]); \
        out_val = bin_##OP(in_val, out_val); \
    } \
\
    s_val[lid] = out_val; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    __local To *s_ptr = s_val + lidy * DIMX; \
\
    if (DIMX == 256) { \
        if (lidx < 128) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx + 128]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
\
    if (DIMX >= 128) { \
        if (lidx <  64) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  64]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
\
    if (DIMX >=  64) { \
        if (lidx <  32) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  32]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
\
    if (lidx < 16) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx + 16]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
\
    if (lidx <  8) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  8]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
\
    if (lidx <  4) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  4]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
\
    if (lidx <  2) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  2]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
\
    if (lidx <  1) s_ptr[lidx] = bin_##OP(s_ptr[lidx], s_ptr[lidx +  1]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
\
    if (cond != 0 && lidx == 0) { \
        oData[groupId_x] = s_ptr[0]; \
    } \
}

INSTANTIATE(ADD_OP)
INSTANTIATE(MUL_OP)
INSTANTIATE(MIN_OP)
INSTANTIATE(MAX_OP)
