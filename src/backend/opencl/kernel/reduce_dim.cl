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

#ifndef THREADS_X
#define THREADS_X 32
#endif

#ifndef DIMY
#define DIMY 8
#endif

#include "ops.cl"

#define INSTANTIATE(OP) \
__kernel __attribute__ ((reqd_work_group_size(32, 8, 1))) \
void reduce_dim_kernel_##OP(__global To *oData, \
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
                            uint groups_x, uint groups_y, uint group_dim, \
                            const dim_type dim, const float init) \
{ \
    const uint lidx = get_local_id(0); \
    const uint lidy = get_local_id(1); \
    const uint lid  = lidy * THREADS_X + lidx; \
 \
    const uint zid = get_group_id(0) / groups_x; \
    const uint wid = get_group_id(1) / groups_y; \
    const uint groupId_x = get_group_id(0) - (groups_x) * zid; \
    const uint groupId_y = get_group_id(1) - (groups_y) * wid; \
    const uint xid = groupId_x * get_local_size(0) + lidx; \
    const uint yid = groupId_y; \
 \
    uint ids[4] = {xid, yid, zid, wid}; \
 \
    const dim_type idims[4] = {idim0, idim1, idim2, idim3}; \
    const dim_type istrides[4] = {istride0, istride1, istride2, istride3}; \
    const dim_type odims[4] = {odim0, odim1, odim2, odim3}; \
 \
    /* There is only one element per group for out \
     * There are get_local_size(1) elements per group for in \
     * Hence increment ids[dim] just after offseting out and before offsetting in */ \
    oData += ids[3] * ostride3 + ids[2] * ostride2 + \
        ids[1] * ostride1 + ids[0] + ooffset; \
    const uint id_dim_out = ids[dim]; \
 \
    ids[dim] = ids[dim] * get_local_size(1) + lidy; \
    iData  += ids[3] * istride3 + ids[2] * istride2 + \
        ids[1] * istride1 + ids[0] + ioffset; \
    const uint id_dim_in = ids[dim]; \
 \
    const uint istride_dim = istrides[dim]; \
 \
    int is_valid = (ids[0] < idim0); \
    is_valid    &= (ids[1] < idim1); \
    is_valid    &= (ids[2] < idim2); \
    is_valid    &= (ids[3] < idim3); \
 \
    __local To s_val[THREADS_X * DIMY]; \
 \
    To out_val = init; \
    for (int id = id_dim_in; is_valid != 0 && (id < idims[dim]); \
         id += group_dim * get_local_size(1)) { \
 \
        To in_val = transform_##OP(*iData); \
        out_val = bin_##OP(in_val, out_val); \
        iData = iData + group_dim * get_local_size(1) * istride_dim; \
    } \
 \
    s_val[lid] = out_val; \
 \
    __local To *s_ptr = s_val + lid; \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (DIMY == 8) { \
        if (lidy < 4) *s_ptr = bin_##OP(*s_ptr, s_ptr[THREADS_X * 4]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMY >= 4) { \
        if (lidy < 2) *s_ptr = bin_##OP(*s_ptr, s_ptr[THREADS_X * 2]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMY >= 2) { \
        if (lidy < 1) *s_ptr = bin_##OP(*s_ptr, s_ptr[THREADS_X * 1]); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (lidy == 0 && is_valid != 0 && \
        (id_dim_out < odims[dim])) { \
        *oData = *s_ptr; \
    } \
 \
}

INSTANTIATE(ADD_OP)
INSTANTIATE(MUL_OP)
INSTANTIATE(MIN_OP)
INSTANTIATE(MAX_OP)
