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

#ifndef SHARED_MEM_SIZE
#define SHARED_MEM_SIZE 256
#endif

#ifndef DIMX
#define DIMX 32
#endif

#ifndef DIMY
#define DIMY 8
#endif

#include "ops.cl"

//#define INSTANTIATE(OP) \
__kernel \
void scan_first_kernel_##OP(__global To *oData, \
                            const dim_type odim0, \
                            const dim_type odim1, \
                            const dim_type odim2, \
                            const dim_type odim3, \
                            const dim_type ostride0, \
                            const dim_type ostride1, \
                            const dim_type ostride2, \
                            const dim_type ostride3, \
                            const dim_type ooffset, \
                            __global To *tData, \
                            const dim_type tstride0, \
                            const dim_type tstride1, \
                            const dim_type tstride2, \
                            const dim_type tstride3, \
                            const dim_type toffset, \
                            const __global Ti *iData, \
                            const dim_type istride0, \
                            const dim_type istride1, \
                            const dim_type istride2, \
                            const dim_type istride3, \
                            const dim_type ioffset, \
                            uint groups_x, uint groups_y, \
                            uint lim, \
                            const To init, \
                            const int isFinalPass) \
{ \
    const int lidx = get_local_id(0); \
    const int lidy = get_local_id(1); \
    const int lid  = lidy * get_local_size(0) + lidx; \
 \
    const int zid = get_group_id(0) / groups_x; \
    const int wid = get_group_id(1) / groups_y; \
    const int groupId_x = get_group_id(0) - (groups_x) * zid; \
    const int groupId_y = get_group_id(1) - (groups_y) * wid; \
    const int xid = groupId_x * get_local_size(0) * lim + lidx; \
    const int yid = groupId_y * get_local_size(1) + lidy; \
 \
    bool cond_yzw = (yid < odim1) && (zid < odim2) && (wid < odim3); \
 \
    iData += wid * istride3 + zid * istride2 + \
        yid * istride1 + ioffset; \
 \
    tData += wid * tstride3 + zid * tstride2 + \
        yid * tstride1 + toffset; \
 \
    oData += wid * ostride3 + zid * ostride2 + \
        yid * ostride1 + ooffset; \
 \
    __local To l_val0[SHARED_MEM_SIZE]; \
    __local To l_val1[SHARED_MEM_SIZE]; \
    __local To *l_val = l_val0; \
    __local To l_tmp[DIMY]; \
 \
    bool flip = 0; \
 \
    const To init_val = init; \
    int id = xid; \
    To val = init_val; \
 \
    const bool isLast = (lidx == (DIMX - 1)); \
 \
    /*if (cond_yzw) {*/ \
 \
        for (int k = 0; k < lim; k++) { \
 \
            if (isLast) l_tmp[lidy] = val; \
 \
            bool cond = ((id < odim0) && cond_yzw); \
            val = cond ? transform_##OP(iData[id]) : init_val; \
            l_val[lid] = val; \
            barrier(CLK_LOCAL_MEM_FENCE); \
 \
            for (int off = 1; off < DIMX; off *= 2) { \
                if (lidx >= off) val = bin_##OP(val, l_val[lid - off]); \
 \
                flip = 1 - flip; \
                l_val = flip ? l_val1 : l_val0; \
                l_val[lid] = val; \
                barrier(CLK_LOCAL_MEM_FENCE); \
            } \
 \
            val = bin_##OP(val, l_tmp[lidy]); \
            if (cond) oData[id] = val; \
            id += DIMX; \
            barrier(CLK_LOCAL_MEM_FENCE); /*FIXME: May be needed only for non nvidia gpus*/ \
        } \
    /*}*/ \
 \
        if (!isFinalPass && isLast && cond_yzw) { \
            tData[groupId_x] = val; \
        } \
} \

#define INSTANTIATE(OP) \
__kernel \
void scan_first_kernel_##OP(__global To *oData, \
                            const dim_type odim0, \
                            const dim_type odim1, \
                            const dim_type odim2, \
                            const dim_type odim3, \
                            const dim_type ostride0, \
                            const dim_type ostride1, \
                            const dim_type ostride2, \
                            const dim_type ostride3, \
                            const dim_type ooffset, \
                            __global To *tData, \
                            const dim_type tstride0, \
                            const dim_type tstride1, \
                            const dim_type tstride2, \
                            const dim_type tstride3, \
                            const dim_type toffset, \
                            const __global Ti *iData, \
                            const dim_type istride0, \
                            const dim_type istride1, \
                            const dim_type istride2, \
                            const dim_type istride3, \
                            const dim_type ioffset, \
                            uint groups_x, uint groups_y, \
                            uint lim, \
                            const To init, \
                            const int isFinalPass) \
{ \
    const int lidx = get_local_id(0); \
    const int lidy = get_local_id(1); \
    const int lid  = lidy * get_local_size(0) + lidx; \
 \
    const int zid = get_group_id(0) / groups_x; \
    const int wid = get_group_id(1) / groups_y; \
    const int groupId_x = get_group_id(0) - (groups_x) * zid; \
    const int groupId_y = get_group_id(1) - (groups_y) * wid; \
    const int xid = groupId_x * get_local_size(0) * lim + lidx; \
    const int yid = groupId_y * get_local_size(1) + lidy; \
 \
    bool cond_yzw  = (yid < odim1); \
         cond_yzw &= (zid < odim2); \
         cond_yzw &= (wid < odim3); \
 \
    iData += wid * istride3 + zid * istride2 + \
        yid * istride1 + ioffset; \
 \
    tData += wid * tstride3 + zid * tstride2 + \
        yid * tstride1 + toffset; \
 \
    oData += wid * ostride3 + zid * ostride2 + \
        yid * ostride1 + ooffset; \
 \
    __local To l_val0[SHARED_MEM_SIZE]; \
    __local To l_val1[SHARED_MEM_SIZE]; \
    __local To *l_val = l_val0; \
    __local To l_tmp[DIMY]; \
 \
    bool flip = 0; \
 \
    const To init_val = init; \
    int id = xid; \
    To val = init_val; \
 \
    const bool isLast = (lidx == (DIMX - 1)); \
 \
    for (int k = 0; k < lim; k++) { \
 \
        if (isLast) l_tmp[lidy] = val; \
 \
        bool cond  = (id < odim0); \
             cond &= (cond_yzw); \
        /*bool cond = ((id < odim0));*/ \
        val = cond ? transform_##OP(iData[id]) : init_val; \
        l_val[lid] = val; \
        barrier(CLK_LOCAL_MEM_FENCE); \
 \
        for (int off = 1; off < DIMX; off *= 2) { \
            if (lidx >= off) val = bin_##OP(val, l_val[lid - off]); \
 \
            flip = 1 - flip; \
            l_val = flip ? l_val1 : l_val0; \
            l_val[lid] = val; \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
 \
        val = bin_##OP(val, l_tmp[lidy]); \
        if (cond) oData[id] = val; \
        id += DIMX; \
        barrier(CLK_LOCAL_MEM_FENCE); /*FIXME: May be needed only for non nvidia gpus*/ \
    } \
 \
    if (!isFinalPass && isLast && cond_yzw) { \
        tData[groupId_x] = val; \
    } \
} \
 \
__kernel \
void bcast_first_kernel_##OP(__global To *oData, \
                             const dim_type odim0, \
                             const dim_type odim1, \
                             const dim_type odim2, \
                             const dim_type odim3, \
                             const dim_type ostride0, \
                             const dim_type ostride1, \
                             const dim_type ostride2, \
                             const dim_type ostride3, \
                             const dim_type ooffset, \
                             const __global To *tData, \
                             const dim_type tstride0, \
                             const dim_type tstride1, \
                             const dim_type tstride2, \
                             const dim_type tstride3, \
                             const dim_type toffset, \
                             uint groups_x, uint groups_y, uint lim) \
{ \
    const int lidx = get_local_id(0); \
    const int lidy = get_local_id(1); \
    const int lid  = lidy * get_local_size(0) + lidx; \
 \
    const int zid = get_group_id(0) / groups_x; \
    const int wid = get_group_id(1) / groups_y; \
    const int groupId_x = get_group_id(0) - (groups_x) * zid; \
    const int groupId_y = get_group_id(1) - (groups_y) * wid; \
    const int xid = groupId_x * get_local_size(0) * lim + lidx; \
    const int yid = groupId_y * get_local_size(1) + lidy; \
 \
    if (groupId_x != 0) { \
        bool cond  = (yid < odim1); \
             cond &= (zid < odim2); \
             cond &= (wid < odim3); \
 \
        if (cond) { \
 \
            tData += wid * tstride3 + zid * tstride2 + \
                yid * tstride1 + toffset; \
 \
            oData += wid * ostride3 + zid * ostride2 + \
                yid * ostride1 + ooffset; \
 \
            To accum = tData[groupId_x - 1]; \
 \
            for (int k = 0, id = xid; \
                 k < lim && id < odim0; \
                 k++, id += DIMX) { \
 \
                oData[id] = bin_##OP(accum, oData[id]); \
            } \
        } \
    } \
}

INSTANTIATE(ADD_OP)
INSTANTIATE(NOTZERO_OP)
