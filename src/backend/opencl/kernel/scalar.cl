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

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1)))
void scalar(__global To *out,
                 const dim_type odim0,
                 const dim_type odim1,
                 const dim_type odim2,
                 const dim_type odim3,
                 const dim_type ostride0,
                 const dim_type ostride1,
                 const dim_type ostride2,
                 const dim_type ostride3,
                 const dim_type ooffset,
                 const Ti scalar)
{
    uint groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    uint threadId = get_local_id(0);

    int idx = groupId * get_local_size(0) * get_local_size(1) + threadId;

    if (idx >= odim3 * ostride3) return;

    To val = (To)scalar;

    out[idx] = val;
}
