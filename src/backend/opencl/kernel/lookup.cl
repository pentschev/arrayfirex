/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef in_t
#define in_t float
#endif

#ifndef idx_t
#define idx_t float
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

__kernel __attribute__ ((reqd_work_group_size(32, 8, 1)))
void lookupND(global in_t * out,
              const dim_type odim0,
              const dim_type odim1,
              const dim_type odim2,
              const dim_type odim3,
              const dim_type ostride0,
              const dim_type ostride1,
              const dim_type ostride2,
              const dim_type ostride3,
              const dim_type ooffset,
              global const in_t * in,
              const dim_type idim0,
              const dim_type idim1,
              const dim_type idim2,
              const dim_type idim3,
              const dim_type istride0,
              const dim_type istride1,
              const dim_type istride2,
              const dim_type istride3,
              const dim_type ioffset,
              global const idx_t * indices,
              dim_type nBBS0,
              dim_type nBBS1,
              dim_type DIM)
{
    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);

    dim_type gz = get_group_id(0)/nBBS0;
    dim_type gw = get_group_id(1)/nBBS1;

    dim_type gx = get_local_size(0) * (get_group_id(0) - gz*nBBS0) + lx;
    dim_type gy = get_local_size(1) * (get_group_id(1) - gw*nBBS1) + ly;

    global const idx_t *idxPtr = indices;

    dim_type i = istride0*(DIM==0 ? trimIndex((dim_type)idxPtr[gx], idim0): gx);
    dim_type j = istride1*(DIM==1 ? trimIndex((dim_type)idxPtr[gy], idim1): gy);
    dim_type k = istride2*(DIM==2 ? trimIndex((dim_type)idxPtr[gz], idim2): gz);
    dim_type l = istride3*(DIM==3 ? trimIndex((dim_type)idxPtr[gw], idim3): gw);

    global const in_t *inPtr = in + (i+j+k+l);
    global in_t *outPtr = out + (gx*ostride0+gy*ostride1+
                                 gz*ostride2+gw*ostride3);

    if (gx<odim0 && gy<odim1 && gz<odim2 && gw<odim3) {
        outPtr[0] = inPtr[0];
    }
}
