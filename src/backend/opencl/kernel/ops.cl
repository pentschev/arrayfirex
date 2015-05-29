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

// NOTZERO_OP
T bin_NOTZERO_OP(T lhs, T rhs)
{
    return lhs + rhs;
}

T transform_NOTZERO_OP(T in)
{
    return (in != 0);
}

// EQ_OP
T bin_EQ_OP(T lhs, T rhs)
{
    return lhs == rhs;
}

To transform_EQ_OP(Ti in)
{
    return(To)(in);
}

// NE_OP
T bin_NE_OP(T lhs, T rhs)
{
    return lhs != rhs;
}

To transform_NE_OP(Ti in)
{
    return(To)(in);
}

// GT_OP
T bin_GT_OP(T lhs, T rhs)
{
    return lhs > rhs;
}

To transform_GT_OP(Ti in)
{
    return(To)(in);
}

// GE_OP
T bin_GE_OP(T lhs, T rhs)
{
    return lhs >= rhs;
}

To transform_GE_OP(Ti in)
{
    return(To)(in);
}

// LT_OP
T bin_LT_OP(T lhs, T rhs)
{
    return lhs < rhs;
}

To transform_LT_OP(Ti in)
{
    return(To)(in);
}

// LE_OP
T bin_LE_OP(T lhs, T rhs)
{
    return lhs <= rhs;
}

To transform_LE_OP(Ti in)
{
    return(To)(in);
}

