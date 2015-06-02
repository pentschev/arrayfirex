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

#include "iops.cl"

#define INSTANTIATE(OP) \
__kernel __attribute__ ((reqd_work_group_size(32, 8, 1))) \
void ireduce_dim_kernel_##OP(__global T *oData, \
                             const dim_type odim0, \
                             const dim_type odim1, \
                             const dim_type odim2, \
                             const dim_type odim3, \
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
                             uint groups_x, uint groups_y, uint group_dim, \
                             const dim_type dim, \
                             const T init, \
                             const int IS_FIRST) \
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
       There are get_local_size(1) elements per group for in \
       Hence increment ids[dim] just after offseting out and before offsetting in */ \
    oData += ids[3] * ostride3 + ids[2] * ostride2 + \
        ids[1] * ostride1 + ids[0] + ooffset; \
    olData += ids[3] * ostride3 + ids[2] * ostride2 + \
        ids[1] * ostride1 + ids[0] + ooffset; \
    const uint id_dim_out = ids[dim]; \
 \
    ids[dim] = ids[dim] * get_local_size(1) + lidy; \
 \
    iData  += ids[3] * istride3 + ids[2] * istride2 + \
        ids[1] * istride1 + ids[0] + ioffset; \
 \
    if (!IS_FIRST) { \
        ilData  += ids[3] * istride3 + ids[2] * istride2 + \
            ids[1] * istride1 + ids[0] + ioffset; \
    } \
 \
    const uint id_dim_in = ids[dim]; \
    const uint istride_dim = istrides[dim]; \
 \
    bool is_valid = \
        (ids[0] < idim0) && \
        (ids[1] < idim1) && \
        (ids[2] < idim2) && \
        (ids[3] < idim3); \
 \
    __local T s_val[THREADS_X * DIMY]; \
    __local T s_idx[THREADS_X * DIMY]; \
 \
    T out_val = init; \
    T out_idx = id_dim_in; \
 \
    if (is_valid && id_dim_in < idims[dim]) { \
        out_val = *iData; \
        if (!IS_FIRST) out_idx = *ilData; \
    } \
 \
    const uint id_dim_in_start = id_dim_in + group_dim * get_local_size(1); \
 \
    for (int id = id_dim_in_start; is_valid && (id < idims[dim]); \
         id += group_dim * get_local_size(1)) { \
 \
        iData = iData + group_dim * get_local_size(1) * istride_dim; \
 \
        if (IS_FIRST != 0) \
            bin_##OP(&out_val, &out_idx, *iData, id); \
        else { \
            ilData = ilData + group_dim * get_local_size(1) * istride_dim; \
            bin_##OP(&out_val, &out_idx, *iData, *ilData); \
        } \
    } \
 \
    s_val[lid] = out_val; \
    s_idx[lid] = out_idx; \
 \
    __local T *s_vptr = s_val + lid; \
    __local T *s_iptr = s_idx + lid; \
    barrier(CLK_LOCAL_MEM_FENCE); \
 \
    if (DIMY == 8) { \
        if (lidy < 4) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[THREADS_X * 4], s_iptr[THREADS_X * 4]); \
            *s_vptr = out_val; \
            *s_iptr = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMY >= 4) { \
        if (lidy < 2) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[THREADS_X * 2], s_iptr[THREADS_X * 2]); \
            *s_vptr = out_val; \
            *s_iptr = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (DIMY >= 2) { \
        if (lidy < 1) { \
            bin_##OP(&out_val, &out_idx, \
                  s_vptr[THREADS_X * 1], s_iptr[THREADS_X * 1]); \
            *s_vptr = out_val; \
            *s_iptr = out_idx; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
 \
    if (lidy == 0 && is_valid && \
        (id_dim_out < odims[dim])) { \
        *oData = *s_vptr; \
        *olData = *s_iptr; \
    } \
 \
}

INSTANTIATE(MIN_OP)
INSTANTIATE(MAX_OP)
