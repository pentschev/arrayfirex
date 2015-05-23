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

dim_type trimIndex(dim_type idx, const dim_type len)
{
    dim_type ret_val = idx;
    dim_type offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

__kernel
void indexKernel(__global T * optr,
                 const dim_type odim0,
                 const dim_type odim1,
                 const dim_type odim2,
                 const dim_type odim3,
                 const dim_type ostride0,
                 const dim_type ostride1,
                 const dim_type ostride2,
                 const dim_type ostride3,
                 __global const T * iptr,
                 const dim_type idim0,
                 const dim_type idim1,
                 const dim_type idim2,
                 const dim_type idim3,
                 const dim_type poff0,
                 const dim_type poff1,
                 const dim_type poff2,
                 const dim_type poff3,
                 const dim_type pstrd0,
                 const dim_type pstrd1,
                 const dim_type pstrd2,
                 const dim_type pstrd3,
                 const char pseq0,
                 const char pseq1,
                 const char pseq2,
                 const char pseq3,
                 __global const T* ptr0, __global const T* ptr1,
                 __global const T* ptr2, __global const T* ptr3,
                 const dim_type nBBS0, const dim_type nBBS1)
{
    // retrive booleans that tell us which indexer to use
    const bool s0 = pseq0;
    const bool s1 = pseq1;
    const bool s2 = pseq2;
    const bool s3 = pseq3;

    const dim_type gz = get_group_id(0)/nBBS0;
    const dim_type gw = get_group_id(1)/nBBS1;
    const dim_type gx = get_local_size(0) * (get_group_id(0) - gz*nBBS0) + get_local_id(0);
    const dim_type gy = get_local_size(1) * (get_group_id(1) - gw*nBBS1) + get_local_id(1);

    if (gx<odim0 && gy<odim1 && gz<odim2 && gw<odim3) {
        // calculate pointer offsets for input
        dim_type i = pstrd0 * trimIndex(s0 ? gx+poff0 : ptr0[gx], idim0);
        dim_type j = pstrd1 * trimIndex(s1 ? gy+poff1 : ptr1[gy], idim1);
        dim_type k = pstrd2 * trimIndex(s2 ? gz+poff2 : ptr2[gz], idim2);
        dim_type l = pstrd3 * trimIndex(s3 ? gw+poff3 : ptr3[gw], idim3);
        // offset input and output pointers
        global const T *src = iptr + (i+j+k+l);
        global T *dst = optr + (gx*ostride0+
                                gy*ostride1+
                                gz*ostride2+
                                gw*ostride3);
        // set the output
        dst[0] = src[0];
    }
}
