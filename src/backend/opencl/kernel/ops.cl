/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// ADD_OP
T bin_ADD_OP(T lhs, T rhs)
{
    return lhs + rhs;
}

To transform_ADD_OP(Ti in)
{
    return(To)(in);
}

// MUL_OP
T bin_MUL_OP(T lhs, T rhs)
{
    return lhs * rhs;
}

To transform_MUL_OP(Ti in)
{
    return(To)(in);
}

// SUB_OP
T bin_SUB_OP(T lhs, T rhs)
{
    return lhs - rhs;
}

To transform_SUB_OP(Ti in)
{
    return(To)(in);
}

// DIV_OP
T bin_DIV_OP(T lhs, T rhs)
{
    return lhs / rhs;
}

To transform_DIV_OP(Ti in)
{
    return(To)(in);
}

// MIN_OP
T bin_MIN_OP(T lhs, T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

T transform_MIN_OP(T in)
{
    return in;
}

// MAX_OP
T bin_MAX_OP(T lhs, T rhs)
{
    return lhs > rhs ? lhs : rhs;
}

T transform_MAX_OP(T in)
{
    return in;
}
