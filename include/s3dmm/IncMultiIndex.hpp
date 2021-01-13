/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

#pragma once

#include "Vec.hpp"
#include <boost/assert.hpp>

namespace s3dmm {

template<unsigned int N, class Scalar>
struct IncMultiIndex
{
    using I = MultiIndex<N, Scalar>;

    // Note: imax is not inclusive!
    static bool inc(I& index, Scalar imax) {
        for (auto d=0u; d<N; ++d) {
            if (index[d]+1 < imax) {
                for (auto d2=0; d2<d; ++d2)
                    index[d2] = 0;
                ++index[d];
                return true;
            }
        }
        return false;
    }

    // Note: imax is not inclusive!
    static bool inc(I& index, const I& imin, const I& imax) {
        for (auto d=0u; d<N; ++d) {
            if (index[d]+1 < imax[d]) {
                for (auto d2=0; d2<d; ++d2)
                    index[d2] = imin[d2];
                ++index[d];
                return true;
            }
        }
        return false;
    }
};

template<class Scalar>
struct IncMultiIndex<1, Scalar>
{
    using I = MultiIndex<1, Scalar>;

    // Note: imax is not inclusive!
    static bool inc(I& index, Scalar imax) {
        if (index[0]+1 < imax) {
            ++index[0];
            return true;
        }
        return false;
    }

    // Note: imax is not inclusive!
    static bool inc(I& index, const I& /*imin*/, const I& imax) {
        if (index[0]+1 < imax[0]) {
            ++index[0];
            return true;
        }
        return false;
    }
};

template<class Scalar>
struct IncMultiIndex<2, Scalar>
{
    using I = MultiIndex<2, Scalar>;

    // Note: imax is not inclusive!
    static bool inc(I& index, unsigned int imax) {
        if (index[0]+1 < imax) {
            ++index[0];
            return true;
        }
        else if (index[1]+1 < imax) {
            index[0] = 0u;
            ++index[1];
            return true;
        }
        return false;
    }

    // Note: imax is not inclusive!
    static bool inc(I& index, const I& imin, const I& imax) {
        if (index[0]+1 < imax[0]) {
            ++index[0];
            return true;
        }
        else if (index[1]+1 < imax[1]) {
            index[0] = imin[0];
            ++index[1];
            return true;
        }
        return false;
    }
};

template<class Scalar>
struct IncMultiIndex<3, Scalar>
{
    using I = MultiIndex<3, Scalar>;

    // Note: imax is not inclusive!
    static bool inc(I& index, Scalar imax) {
        if (index[0]+1 < imax) {
            ++index[0];
            return true;
        }
        else if (index[1]+1 < imax) {
            index[0] = 0u;
            ++index[1];
            return true;
        }
        else if (index[2]+1 < imax) {
            index[0] = index[1] = 0u;
            ++index[2];
            return true;
        }
        return false;
    }

    // Note: imax is not inclusive!
    static bool inc(I& index, const I& imin, const I& imax) {
        if (index[0]+1 < imax[0]) {
            ++index[0];
            return true;
        }
        else if (index[1]+1 < imax[1]) {
            index[0] = imin[0];
            ++index[1];
            return true;
        }
        else if (index[2]+1 < imax[2]) {
            index[0] = imin[0];
            index[1] = imin[1];
            ++index[2];
            return true;
        }
        return false;
    }
};



template<unsigned int N, class Scalar>
struct Inc01MultiIndex
{
    static bool inc(MultiIndex<N, Scalar>& index) {
        for (auto d=0u; d<N; ++d) {
            if (index[d] == 0u) {
                for (auto d2=0; d2<d; ++d2)
                    index[d2] = 0;
                ++index[d];
                return true;
            }
        }
        return false;
    }

    static bool inc(MultiIndex<N, Scalar>& index, const MultiIndex<N, Scalar>& imax) {
        for (auto d=0u; d<N; ++d) {
            if (index[d] == 0u) {
                for (auto d2=0; d2<d; ++d2)
                    index[d2] = 0;
                index[d] = imax[d];
                return true;
            }
        }
        return false;
    }
};

template<class Scalar>
struct Inc01MultiIndex<1, Scalar>
{
    static bool inc(MultiIndex<1, Scalar>& index) {
        if (index[0] == 0u) {
            ++index[0];
            return true;
        }
        return false;
    }

    static bool inc(MultiIndex<1, Scalar>& index, const MultiIndex<1, Scalar>& imax) {
        if (index[0] == 0u) {
            index[0] = imax[0];
            return true;
        }
        return false;
    }
};

template<class Scalar>
struct Inc01MultiIndex<2, Scalar>
{
    static bool inc(MultiIndex<2, Scalar>& index) {
        if (index[0] == 0u) {
            ++index[0];
            return true;
        }
        else if (index[1] == 0u) {
            index[0] = 0u;
            ++index[1];
            return true;
        }
        return false;
    }

    static bool inc(MultiIndex<2, Scalar>& index, const MultiIndex<2, Scalar>& imax) {
        if (index[0] == 0u) {
            index[0] = imax[0];
            return true;
        }
        else if (index[1] == 0u) {
            index[0] = 0u;
            index[1] = imax[1];
            return true;
        }
        return false;
    }
};

template<class Scalar>
struct Inc01MultiIndex<3, Scalar>
{
    static bool inc(MultiIndex<3, Scalar>& index) {
        if (index[0] == 0u) {
            ++index[0];
            return true;
        }
        else if (index[1] == 0u) {
            index[0] = 0u;
            ++index[1];
            return true;
        }
        else if (index[2] == 0u) {
            index[0] = index[1] = 0u;
            ++index[2];
            return true;
        }
        return false;
    }
    static bool inc(MultiIndex<3, Scalar>& index, const MultiIndex<3, Scalar>& imax) {
        if (index[0] == 0u) {
            index[0] = imax[0];
            return true;
        }
        else if (index[1] == 0u) {
            index[0] = 0u;
            index[1] = imax[1];
            return true;
        }
        else if (index[2] == 0u) {
            index[0] = index[1] = 0u;
            index[2] = imax[2];
            return true;
        }
        return false;
    }
};

template<unsigned int N, class Scalar>
inline bool incMultiIndex(MultiIndex<N, Scalar>& index, Scalar imax) {
    return IncMultiIndex<N, Scalar>::inc(index, imax);
}

template<unsigned int N, class Scalar>
inline bool incMultiIndex(
        MultiIndex<N, Scalar>& index,
        const MultiIndex<N, Scalar>& imin,
        const MultiIndex<N, Scalar>& imax)
{
    return IncMultiIndex<N, Scalar>::inc(index, imin, imax);
}

template<unsigned int N, class Scalar>
inline bool inc01MultiIndex(MultiIndex<N, Scalar>& index) {
    return Inc01MultiIndex<N, Scalar>::inc(index);
}

template<unsigned int N, class Scalar>
inline bool inc01MultiIndex(MultiIndex<N, Scalar>& index, const MultiIndex<N, Scalar>& imax) {
    return Inc01MultiIndex<N, Scalar>::inc(index, imax);
}

} // s3dmm

