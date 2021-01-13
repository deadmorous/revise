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
#include "linsolv_cramer.hpp"

namespace s3dmm {

template<>
class ElementApprox<2, MeshElementType::Quad>
{
private:
    using Vec2 = Vec<2, real_type>;

public:
    template<class ElementNodes>
    explicit ElementApprox(const ElementNodes& elementNodalData)
    {
        auto it = elementNodalData.begin();
        m_fieldsPerNode = it->size() - 2;
        m_x0 = (it++)->begin();
        m_x1 = (it++)->begin();
        m_x3 = (it++)->begin();
        m_x2 = (it++)->begin();
        auto& r0 = *reinterpret_cast<const Vec2*>(m_x0);
        auto& r1 = *reinterpret_cast<const Vec2*>(m_x1);
        auto& r2 = *reinterpret_cast<const Vec2*>(m_x2);
        auto& r3 = *reinterpret_cast<const Vec2*>(m_x3);
        constexpr const real_type _025 = make_real(0.25);
        m_A = _025*(r0+r1+r2+r3);
        *reinterpret_cast<Vec2*>(m_B + 0) = _025*(-r0+r1-r2+r3);
        *reinterpret_cast<Vec2*>(m_B + 2) = _025*(-r0-r1+r2+r3);
        m_C = _025*(r0-r1-r2+r3);
        linsolv_crammer::invert2x2(m_iB, m_B);
        m_size = std::accumulate(m_B, m_B+4, make_real(0), [](real_type a, real_type b) { return a + std::abs(b); });
    }

    std::pair<bool, bool> param(Vec2& param, const Vec2& x) const
    {
        // Compute linear estimation of coordinates xi, eta
        auto x_A = x - m_A;
        auto xi  = m_iB[0]*x_A[0] + m_iB[2]*x_A[1];
        auto eta = m_iB[1]*x_A[0] + m_iB[3]*x_A[1];

        constexpr const real_type cullThreshold = make_real(2);
        if (std::abs(xi) > cullThreshold || std::abs(eta) > cullThreshold)
            return {false, true};

        auto isInside = [&xi, &eta]() {
            constexpr const real_type margin = make_real(1 + 1e-4);
            return std::abs(xi) <= margin && std::abs(eta) <= margin;
        };

        auto initiallyInside = isInside();

        // Run Newton's method to adjust xi, eta
        constexpr const unsigned int MaxNewtonSteps = 10;
        const real_type newtonFTol = make_real(1e-5) * m_size;
        Vec2 F = m_C*(-xi*eta);

        for (auto iNewton=0u; ; ++iNewton) {
            if (std::abs(F[0]) + std::abs(F[1]) <= newtonFTol)
                break;
            if (iNewton > MaxNewtonSteps)
                return {initiallyInside, false};
                // throw std::runtime_error("ElementApprox<2, MeshElementType::Quad>: Newton's method did not converge");
            real_type g[4];
            g[0] = m_B[0] + m_C[0]*eta;
            g[1] = m_B[1] + m_C[1]*eta;
            g[2] = m_B[2] + m_C[0]*xi;
            g[3] = m_B[3] + m_C[1]*xi;
            real_type dy[2];
            linsolv_crammer::solve2x2(dy, g, F.data());
            xi  += dy[0];
            eta += dy[1];
            F = -m_C*(dy[0]*dy[1]);
        }

        constexpr const real_type margin = make_real(1 + 1e-4);
        if (isInside()) {
            param[0] = xi;
            param[1] = eta;
            return {true, true};
        }
        else
            return {false, true};
    }

    bool operator()(real_type *field, const Vec2& x) const
    {
        Vec2 p;
        auto pr = param(p, x);
        if (pr.second && pr.first) {
            constexpr const real_type _025 = make_real(0.25);
            constexpr const real_type _1 = make_real(1);
            auto _1_plus_xi = _1 + p[0];
            auto _1_minus_xi = _1 - p[0];
            auto _1_plus_eta = _1 + p[1];
            auto _1_minus_eta = _1 - p[1];
            auto N0 = _025*_1_minus_xi*_1_minus_eta;
            auto N1 = _025*_1_plus_xi *_1_minus_eta;
            auto N2 = _025*_1_minus_xi*_1_plus_eta;
            auto N3 = _025*_1_plus_xi *_1_plus_eta;
            for (auto i=0u; i<m_fieldsPerNode; ++i)
                field[i] = N0*m_x0[2+i] + N1*m_x1[2+i] + N2*m_x2[2+i] + N3*m_x3[2+i];
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
        constexpr const real_type _025 = make_real(0.25);
        constexpr const real_type _1 = make_real(1);
        auto _1_plus_xi = _1 + p[0];
        auto _1_minus_xi = _1 - p[0];
        auto _1_plus_eta = _1 + p[1];
        auto _1_minus_eta = _1 - p[1];
        auto N0 = _025*_1_minus_xi*_1_minus_eta;
        auto N1 = _025*_1_plus_xi *_1_minus_eta;
        auto N2 = _025*_1_minus_xi*_1_plus_eta;
        auto N3 = _025*_1_plus_xi *_1_plus_eta;

        auto it = elementNodalData.begin();
        auto fieldsPerNode = it->size();
        auto nf0 = (it++)->begin();
        auto nf1 = (it++)->begin();
        auto nf3 = (it++)->begin();
        auto nf2 = (it++)->begin();
        for (auto i=0u; i<fieldsPerNode; ++i)
            field[i] = N0*nf0[i] + N1*nf1[i] + N2*nf2[i] + N3*nf3[i];
    }

private:
    unsigned int m_fieldsPerNode;
    const real_type *m_x0;
    const real_type *m_x1;
    const real_type *m_x2;
    const real_type *m_x3;
    Vec2 m_A;
    real_type m_B[4];   // columnwise 2x2 matrix
    real_type m_iB[4];  // columnwise 2x2 matrix
    Vec2 m_C;
    real_type m_size;   // Element characteristic size
};

} // s3dmm
