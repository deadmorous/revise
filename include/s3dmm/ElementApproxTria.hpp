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

#include "ElementApproxFwd.hpp"

namespace s3dmm {

template<>
class ElementApprox<2, MeshElementType::Triangle>
{
private:
    using Vec2 = Vec<2, real_type>;

public:
    ElementApprox() = default;  // Need to use arrays of approximators

    template<class ElementNodes>
    explicit ElementApprox(const ElementNodes& elementNodalData)
    {
        auto it = elementNodalData.begin();
        m_fieldsPerNode = it->size() - 2;
        m_x0 = (it++)->begin();
        m_x1 = (it++)->begin();
        m_x2 = (it++)->begin();
        auto& r0 = *reinterpret_cast<const Vec2*>(m_x0);
        auto& r1 = *reinterpret_cast<const Vec2*>(m_x1);
        auto& r2 = *reinterpret_cast<const Vec2*>(m_x2);
        auto r01 = r1 - r0;
        auto r02 = r2 - r0;
        m_N1 = rot_cw(r02);
        m_N2 = rot_ccw(r01);
        auto ivol = make_real(1) / (r01 % r02);
        m_N1 *= ivol;
        m_N2 *= ivol;
    }

    std::pair<bool, bool> param(Vec2& param, const Vec2& x) const
    {
        // Compute L-coordinates L1, L2 (L0=1-L1-L2)
        auto& r0 = *reinterpret_cast<const Vec2*>(m_x0);
        auto r0x = x - r0;
        auto L1 = r0x * m_N1;
        auto L2 = r0x * m_N2;
        if (L1 < 0 || L2 < 0)
            return {false, true};
        auto L0 = make_real(1) - (L1 + L2);
        if (L0 < 0)
            return {false, true};
        param = {L1, L2};
        return {true, true};
    }

    bool operator()(real_type *field, const Vec2& x) const
    {
        Vec2 p;
        auto pr = param(p, x);
        if (pr.second && pr.first) {
            auto L1 = p[0];
            auto L2 = p[1];
            auto L0 = make_real(1) - (L1 + L2);
            for (auto i=0u; i<m_fieldsPerNode; ++i)
                field[i] = L0*m_x0[2+i] + L1*m_x1[2+i] + L2*m_x2[2+i];
            return true;
        }
        else
            return false;
    }

    template<class ElementNodes>
    static void approx(
            real_type *field,
            const ElementNodes& elementNodalData,
            const Vec2& p)
    {
        auto L1 = p[0];
        auto L2 = p[1];
        auto L0 = make_real(1) - (L1 + L2);

        auto it = elementNodalData.begin();
        auto fieldsPerNode = it->size();
        auto nf0 = (it++)->begin();
        auto nf1 = (it++)->begin();
        auto nf2 = (it++)->begin();
        for (auto i=0u; i<fieldsPerNode; ++i)
            field[i] = L0*nf0[i] + L1*nf1[i] + L2*nf2[i];
    }

private:
    unsigned int m_fieldsPerNode;
    const real_type *m_x0;
    const real_type *m_x1;
    const real_type *m_x2;
    Vec2 m_N1;
    Vec2 m_N2;
};

} // s3dmm
