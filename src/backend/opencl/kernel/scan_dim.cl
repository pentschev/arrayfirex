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

#ifndef To
#define To float
#endif

#ifndef Ti
#define Ti float
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
__kernel \
void scan_dim_kernel_##OP(__global To *oData, \
                          const dim_type odim0, \
                          const dim_type odim1, \
                          const dim_type odim2, \
                          const dim_type odim3, \
                          const dim_type ostride0, \
                          const dim_type ostride1, \
                          const dim_type ostride2, \
                          const dim_type ostride3, \
                          __global To *tData, \
                          const dim_type tdim0, \
                          const dim_type tdim1, \
                          const dim_type tdim2, \
                          const dim_type tdim3, \
                          const dim_type tstride0, \
                          const dim_type tstride1, \
                          const dim_type tstride2, \
                          const dim_type tstride3, \
                          const __global Ti *iData, \
                          const dim_type istride0, \
                          const dim_type istride1, \
                          const dim_type istride2, \
                          const dim_type istride3, \
                          uint groups_x, \
                          uint groups_y, \
                          uint groups_dim, \
                          uint lim, \
                          const int dim, \
                          const To init, \
                          const int isFinalPass) \
{ \
    const int lidx = get_local_id(0); \
    const int lidy = get_local_id(1); \
    const int lid  = lidy * THREADS_X + lidx; \
 \
    const int zid = get_group_id(0) / groups_x; \
    const int wid = get_group_id(1) / groups_y; \
    const int groupId_x = get_group_id(0) - (groups_x) * zid; \
    const int groupId_y = get_group_id(1) - (groups_y) * wid; \
    const int xid = groupId_x * get_local_size(0) + lidx; \
    const int yid = groupId_y; \
 \
    int ids[4] = {xid, yid, zid, wid}; \
    int odims[4] = {odim0, odim1, odim2, odim3}; \
    int tdims[4] = {tdim0, tdim1, tdim2, tdim3}; \
    int ostrides[4] = {ostride0, ostride1, ostride2, ostride3}; \
    int istrides[4] = {istride0, istride1, istride2, istride3}; \
 \
    /* There is only one element per group for out \
       There are DIMY elements per group for in \
       Hence increment ids[dim] just after offseting out and before offsetting in */ \
    tData += ids[3] * tstride3 + ids[2] * tstride2 + ids[1] * tstride1 + ids[0]; \
    const int groupId_dim = ids[dim]; \
 \
    ids[dim] = ids[dim] * DIMY * lim + lidy; \
    oData  += ids[3] * ostride3 + ids[2] * ostride2 + ids[1] * ostride1 + ids[0]; \
    iData  += ids[3] *  istride3 + ids[2] *  istride2 + ids[1] *  istride1 + ids[0]; \
    int id_dim = ids[dim]; \
    const int out_dim = odims[dim]; \
 \
    bool is_valid  = (ids[0] < odim0); \
         is_valid &= (ids[1] < odim1); \
         is_valid &= (ids[2] < odim2); \
         is_valid &= (ids[3] < odim3); \
 \
    const int ostride_dim = ostrides[dim]; \
    const int istride_dim =  istrides[dim]; \
 \
    __local To l_val0[THREADS_X * DIMY]; \
    __local To l_val1[THREADS_X * DIMY]; \
    __local To *l_val = l_val0; \
    __local To l_tmp[THREADS_X]; \
 \
    bool flip = 0; \
    const To init_val  = init; \
    To val = init_val; \
    const bool isLast = (lidy == (DIMY - 1)); \
 \
    for (int k = 0; k < lim; k++) { \
 \
        if (isLast) l_tmp[lidx] = val; \
 \
        bool cond  = (is_valid); \
             cond &= (id_dim < out_dim); \
        val = cond ? transform_##OP(*iData) : init_val; \
        l_val[lid] = val; \
        barrier(CLK_LOCAL_MEM_FENCE); \
 \
        int start = 0; \
        for (int off = 1; off < DIMY; off *= 2) { \
 \
            if (lidy >= off) val = bin_##OP(val, l_val[lid - off * THREADS_X]); \
 \
            flip = 1 - flip; \
            l_val = flip ? l_val1 : l_val0; \
            l_val[lid] = val; \
 \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
 \
        val = bin_##OP(val, l_tmp[lidx]); \
        if (cond) *oData = val; \
        barrier(CLK_LOCAL_MEM_FENCE); \
 \
        id_dim += DIMY; \
        iData += DIMY * istride_dim; \
        oData += DIMY * ostride_dim; \
    } \
 \
    if (!isFinalPass && \
        is_valid && \
        (groupId_dim < tdims[dim]) && \
        isLast) { \
        *tData = val; \
    } \
} \
 \
__kernel \
void bcast_dim_kernel_##OP(__global To *oData, \
                           const dim_type odim0, \
                           const dim_type odim1, \
                           const dim_type odim2, \
                           const dim_type odim3, \
                           const dim_type ostride0, \
                           const dim_type ostride1, \
                           const dim_type ostride2, \
                           const dim_type ostride3, \
                           const __global To *tData, \
                           const dim_type tstride0, \
                           const dim_type tstride1, \
                           const dim_type tstride2, \
                           const dim_type tstride3, \
                           uint groups_x, \
                           uint groups_y, \
                           uint groups_dim, \
                           uint lim, \
                           int dim) \
{ \
    const int lidx = get_local_id(0); \
    const int lidy = get_local_id(1); \
    const int lid  = lidy * THREADS_X + lidx; \
 \
    const int zid = get_group_id(0) / groups_x; \
    const int wid = get_group_id(1) / groups_y; \
    const int groupId_x = get_group_id(0) - (groups_x) * zid; \
    const int groupId_y = get_group_id(1) - (groups_y) * wid; \
    const int xid = groupId_x * get_local_size(0) + lidx; \
    const int yid = groupId_y; \
 \
    int ids[4] = {xid, yid, zid, wid}; \
    int odims[4] = {odim0, odim1, odim2, odim3}; \
    int ostrides[4] = {ostride0, ostride1, ostride2, ostride3}; \
    int tstrides[4] = {tstride0, tstride1, tstride2, tstride3}; \
    const int groupId_dim = ids[dim]; \
 \
    if (groupId_dim == 0) return; \
 \
    /* There is only one element per group for out \
       There are DIMY elements per group for in \
       Hence increment ids[dim] just after offseting out and before offsetting in */ \
    tData += ids[3] * tstride3 + ids[2] * tstride2 + ids[1] * tstride1 + ids[0]; \
 \
    ids[dim] = ids[dim] * DIMY * lim + lidy; \
    oData  += ids[3] * ostride3 + ids[2] * ostride2 + ids[1] * ostride1 + ids[0]; \
 \
    const int id_dim = ids[dim]; \
    const int out_dim = odims[dim]; \
 \
    bool is_valid  = (ids[0] < odim0); \
         is_valid &= (ids[1] < odim1); \
         is_valid &= (ids[2] < odim2); \
         is_valid &= (ids[3] < odim3); \
 \
    if (!is_valid) return; \
 \
    To accum = *(tData - tstrides[dim]); \
 \
    const int ostride_dim = ostrides[dim]; \
 \
    for (int k = 0, id = id_dim; \
         is_valid && k < lim && (id < out_dim); \
         k++, id += DIMY) { \
 \
        *oData = bin_##OP(*oData, accum); \
        oData += DIMY * ostride_dim; \
    } \
}

INSTANTIATE(ADD_OP)
INSTANTIATE(NOTZERO_OP)
