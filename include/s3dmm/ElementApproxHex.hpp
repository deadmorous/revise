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
class ElementApprox<3, MeshElementType::Hexahedron>
{
private:
    using Vec3 = Vec<3, real_type>;

public:
    template<class ElementNodes>
    explicit ElementApprox(const ElementNodes& elementNodalData)
    {
        auto it = elementNodalData.begin();
        m_fieldsPerNode = it->size() - 3;
        m_x0 = (it++)->begin();
        m_x1 = (it++)->begin();
        m_x3 = (it++)->begin();
        m_x2 = (it++)->begin();
        m_x4 = (it++)->begin();
        m_x5 = (it++)->begin();
        m_x7 = (it++)->begin();
        m_x6 = (it++)->begin();
        auto& r0 = *reinterpret_cast<const Vec3*>(m_x0);
        auto& r1 = *reinterpret_cast<const Vec3*>(m_x1);
        auto& r2 = *reinterpret_cast<const Vec3*>(m_x2);
        auto& r3 = *reinterpret_cast<const Vec3*>(m_x3);
        auto& r4 = *reinterpret_cast<const Vec3*>(m_x4);
        auto& r5 = *reinterpret_cast<const Vec3*>(m_x5);
        auto& r6 = *reinterpret_cast<const Vec3*>(m_x6);
        auto& r7 = *reinterpret_cast<const Vec3*>(m_x7);
        constexpr const real_type _0125 = make_real(0.125);
        m_A = _0125*(r0+r1+r2+r3+r4+r5+r6+r7);
        *reinterpret_cast<Vec3*>(m_B + 0) = _0125*(-r0+r1-r2+r3-r4+r5-r6+r7);
        *reinterpret_cast<Vec3*>(m_B + 3) = _0125*(-r0-r1+r2+r3-r4-r5+r6+r7);
        *reinterpret_cast<Vec3*>(m_B + 6) = _0125*(-r0-r1-r2-r3+r4+r5+r6+r7);
        m_C   = _0125*(-r0+r1+r2-r3+r4-r5-r6+r7);
        m_Cxe = _0125*( r0-r1-r2+r3+r4-r5-r6+r7);
        m_Cez = _0125*( r0+r1-r2-r3-r4-r5+r6+r7);
        m_Czx = _0125*( r0-r1+r2-r3-r4+r5-r6+r7);
        linsolv_crammer::invert3x3(m_iB, m_B);
        m_size = make_real(2./3) * std::accumulate(m_B, m_B+9, make_real(0), [](real_type a, real_type b) { return a + std::abs(b); });
    }

    std::pair<bool, bool> param(Vec3& param, const Vec3& x) const
    {
        // Compute linear estimation of coordinates xi, eta
        auto x_A = x - m_A;
        auto xi   = m_iB[0]*x_A[0] + m_iB[3]*x_A[1] + m_iB[6]*x_A[2];
        auto eta  = m_iB[1]*x_A[0] + m_iB[4]*x_A[1] + m_iB[7]*x_A[2];
        auto zeta = m_iB[2]*x_A[0] + m_iB[5]*x_A[1] + m_iB[8]*x_A[2];

        constexpr const real_type cullThreshold = make_real(2);
        if (std::abs(xi) > cullThreshold || std::abs(eta) > cullThreshold || std::abs(zeta) > cullThreshold)
            return {false, true};

        auto isInside = [&xi, &eta, &zeta]() {
            constexpr const real_type margin = make_real(1 + 1e-3);
            return std::abs(xi) <= margin && std::abs(eta) <= margin && std::abs(zeta) <= margin;
        };

        auto initiallyInside = isInside();

        // Run Newton's method to adjust xi, eta
        constexpr const unsigned int MaxNewtonSteps = 10;
        const real_type newtonFTol = make_real(1e-5) * m_size;
        Vec3 F = m_Cxe*(-xi*eta) - m_Cez*(eta*zeta) - m_Czx*(zeta*xi) - m_C*(xi*eta*zeta);
        for (auto iNewton=0u; ; ++iNewton) {
            if (std::abs(F[0]) + std::abs(F[1]) + std::abs(F[2]) <= newtonFTol)
                break;
            if (iNewton > MaxNewtonSteps)
                return {initiallyInside, false};
                // throw std::runtime_error("ElementApprox<3, MeshElementType::Hexahedron>: Newton's method did not converge");
            real_type g[9];
            *reinterpret_cast<Vec3*>(g+0) = *reinterpret_cast<const Vec3*>(m_B+0) + m_Cxe*eta + m_Czx*zeta + m_C*(eta*zeta);
            *reinterpret_cast<Vec3*>(g+3) = *reinterpret_cast<const Vec3*>(m_B+3) + m_Cxe*xi  + m_Cez*zeta + m_C*(xi*zeta);
            *reinterpret_cast<Vec3*>(g+6) = *reinterpret_cast<const Vec3*>(m_B+6) + m_Czx*xi  + m_Cez*eta  + m_C*(xi*eta);
            real_type dy[3];
            linsolv_crammer::solve3x3(dy, g, F.data());
            auto dxe = dy[0]*dy[1];
            auto dez = dy[1]*dy[2];
            auto dzx = dy[2]*dy[0];
            auto dxez = dxe*dy[2];
            F = m_Cxe*(-dxe) - m_Cez*dez - m_Czx*dzx - m_C*(xi*dez + eta*dzx + zeta*dxe + dxez);
            xi   += dy[0];
            eta  += dy[1];
            zeta += dy[2];
        }

        if (isInside()) {
            param[0] = xi;
            param[1] = eta;
            param[2] = zeta;
            return {true, true};
        }
        else
            return {false, true};
    }

    bool operator()(real_type *field, const Vec3& x) const
    {
        Vec3 p;
        auto pr = param(p, x);
        if (pr.second && pr.first) {
            constexpr const real_type _0125 = make_real(0.25);
            constexpr const real_type _1 = make_real(1);
            auto _1_plus_xi = _1 + p[0];
            auto _1_minus_xi = _1 - p[0];
            auto _1_plus_eta = _1 + p[1];
            auto _1_minus_eta = _1 - p[1];
            auto _1_plus_zeta = _1 + p[2];
            auto _1_minus_zeta = _1 - p[2];
            auto N0 = _0125*_1_minus_xi*_1_minus_eta;
            auto N1 = _0125*_1_plus_xi *_1_minus_eta;
            auto N2 = _0125*_1_minus_xi*_1_plus_eta;
            auto N3 = _0125*_1_plus_xi *_1_plus_eta;
            auto N4 = N0*_1_plus_zeta;
            auto N5 = N1*_1_plus_zeta;
            auto N6 = N2*_1_plus_zeta;
            auto N7 = N3*_1_plus_zeta;
            N0 *= _1_minus_zeta;
            N1 *= _1_minus_zeta;
            N2 *= _1_minus_zeta;
            N3 *= _1_minus_zeta;
            for (auto i=0u; i<m_fieldsPerNode; ++i)
                field[i] =
                        N0*m_x0[3+i] + N1*m_x1[3+i] + N2*m_x2[3+i] + N3*m_x3[3+i] +
                        N4*m_x4[3+i] + N5*m_x5[3+i] + N6*m_x6[3+i] + N7*m_x7[3+i];
            return true;
        }
        else
            return false;
    }

    template<class ElementNodes>
    static void approx(
            real_type *field,
            const ElementNodes& elementNodalData,
            const Vec3& p)
    {
        constexpr const real_type _0125 = make_real(0.25);
        constexpr const real_type _1 = make_real(1);
        auto _1_plus_xi = _1 + p[0];
        auto _1_minus_xi = _1 - p[0];
        auto _1_plus_eta = _1 + p[1];
        auto _1_minus_eta = _1 - p[1];
        auto _1_plus_zeta = _1 + p[2];
        auto _1_minus_zeta = _1 - p[2];
        auto N0 = _0125*_1_minus_xi*_1_minus_eta;
        auto N1 = _0125*_1_plus_xi *_1_minus_eta;
        auto N2 = _0125*_1_minus_xi*_1_plus_eta;
        auto N3 = _0125*_1_plus_xi *_1_plus_eta;
        auto N4 = N0*_1_plus_zeta;
        auto N5 = N1*_1_plus_zeta;
        auto N6 = N2*_1_plus_zeta;
        auto N7 = N3*_1_plus_zeta;
        N0 *= _1_minus_zeta;
        N1 *= _1_minus_zeta;
        N2 *= _1_minus_zeta;
        N3 *= _1_minus_zeta;

        auto it = elementNodalData.begin();
        auto fieldsPerNode = it->size();
        auto nf0 = (it++)->begin();
        auto nf1 = (it++)->begin();
        auto nf3 = (it++)->begin();
        auto nf2 = (it++)->begin();
        auto nf4 = (it++)->begin();
        auto nf5 = (it++)->begin();
        auto nf7 = (it++)->begin();
        auto nf6 = (it++)->begin();
        for (auto i=0u; i<fieldsPerNode; ++i)
            field[i] =
                    N0*nf0[i] + N1*nf1[i] + N2*nf2[i] + N3*nf3[i] +
                    N4*nf4[i] + N5*nf5[i] + N6*nf6[i] + N7*nf7[i];
    }

private:
    unsigned int m_fieldsPerNode;
    const real_type *m_x0;
    const real_type *m_x1;
    const real_type *m_x2;
    const real_type *m_x3;
    const real_type *m_x4;
    const real_type *m_x5;
    const real_type *m_x6;
    const real_type *m_x7;
    Vec3 m_A;
    real_type m_B[9];   // columnwise 3x3 matrix
    real_type m_iB[9];  // columnwise 3x3 matrix
    Vec3 m_C;
    Vec3 m_Cxe;
    Vec3 m_Cez;
    Vec3 m_Czx;
    real_type m_size;   // Element characteristic size
};

} // s3dmm
