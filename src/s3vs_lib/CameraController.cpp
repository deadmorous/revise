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

#include "CameraController.hpp"

using namespace s3dmm;

namespace s3vs
{

using vlVec3r = vl::Vector3<real_type>;

namespace {

template <class T>
inline const vl::Vector3<T>& asVl(const s3dmm::Vec<3, T>& x) {
    return *reinterpret_cast<const vl::Vector3<T>*>(&x);
}

template <class T>
inline vl::Vector3<T>& asVl(s3dmm::Vec<3, T>& x) {
    return *reinterpret_cast<vl::Vector3<T>*>(&x);
}

inline Matrix3r outerProduct(const vlVec3r& a, const vlVec3r& b)
{
    return Matrix3r(
            a[0]*b[0], a[0]*b[1], a[0]*b[2],
            a[1]*b[0], a[1]*b[1], a[1]*b[2],
            a[2]*b[0], a[2]*b[1], a[2]*b[2]);
}

inline Matrix3r spinMatrix(const vlVec3r& s)
{
    return Matrix3r(
            0,   -s[2],  s[1],
            s[2], 0,    -s[0],
           -s[1], s[0],  0);
}

inline Matrix3r rotationMatrix(const vlVec3r& rotationVector)
{
    auto fi = rotationVector.length();
    if( fi > std::numeric_limits<real_type>::epsilon() ) {
        auto n1 = rotationVector / fi;
        auto c = cos(fi);
        auto s = sin(fi);
        return outerProduct(n1, n1*(1-c)) + Matrix3r(c) + spinMatrix(n1*s);
    }
    else
        return Matrix3r() + spinMatrix(rotationVector);
}

inline vlVec3r otherDirVector(const vlVec3r& x)
{
    auto x0 = std::abs(x[0]);
    auto x1 = std::abs(x[1]);
    auto x2 = std::abs(x[2]);
    auto index = x0 < x1? (x0 < x2? 0: 2): (x1 < x2? 1: 2);
    vlVec3r result;
    result[index] = make_real(1);
    return result;
}

} // anonymous namespace

CameraController::CameraController()
{
    resetCameraTransform();
}

void CameraController::resetCameraTransform()
{
    vl::Vector3<real_type> eyeDir(2, 1, 3);
    eyeDir.normalize();
    const auto CubeSize = make_real(10);
    const auto Margin = make_real(0.01);
    auto D = CubeSize * sqrt(make_real(3)) *(make_real(1) + Margin);
    auto L = D / (make_real(2)*make_real(sin(0.5*m_fovY*M_PI/180)));
    auto eye = eyeDir * L;
    vl::Vector3<real_type> center(0, 0, 0);
    vl::Vector3<real_type> up(0, 1, 0);
    m_cameraTransform = Matrix4r::getLookAt(eye, center, up);
    m_centerPosition = { 0, 0, 0 };
}

void CameraController::setOptions(const Options& options)
{
    m_options = options;
    if (m_options.respectUpDirection)
        computeCurrentRotation();
}

auto CameraController::options() const -> Options {
    return m_options;
}

void CameraController::setViewportSize(const Vec2u& viewportSize) {
    m_viewportSize = viewportSize;
}

void CameraController::setCameraTransform(const Matrix4r& cameraTransform) {
    m_cameraTransform = cameraTransform;
}

void CameraController::setCenterPosition(const Vec3r& centerPosition) {
    m_centerPosition = centerPosition;
}

void CameraController::setFovY(real_type fovY) {
    m_fovY = fovY;
}

Vec2u CameraController::viewportSize() const {
    return m_viewportSize;
}

Matrix4r CameraController::cameraTransform() const {
    return m_cameraTransform;
}

Vec3r CameraController::centerPosition() const {
    return m_centerPosition;
}

real_type CameraController::fovY() const {
    return m_fovY;
}

void CameraController::rotate(const Vec2r& delta)
{
    auto vfactor = m_options.rotationSensitivity * make_real(M_PI/2) / (m_viewportSize[1]);
    auto v = delta * vfactor;
    auto te = m_cameraTransform.getT();
    auto tcami = m_cameraTransform.getInverse();
    auto re = -tcami.get3x3()*te;
    auto e1e = tcami.getX();
    e1e.normalize();
    auto e2e = tcami.getY();
    e2e.normalize();
    auto theta = v[1]*e1e - v[0]*e2e;
    auto dP = rotationMatrix(theta);
    auto& rc = asVl(m_centerPosition);
    auto re2 = rc + dP * (re - rc);
    auto P2 = m_cameraTransform.get3x3() * dP.getTransposed();
    m_cameraTransform.set3x3(P2);
    m_cameraTransform.setT(-P2*re2);
    if (m_options.respectUpDirection)
        computeCurrentRotation();
}

void CameraController::rotateInPlane(const Vec2r& delta)
{
    BOOST_ASSERT(false); // TODO
}

void CameraController::pan(const Vec2r& delta)
{
    auto te = m_cameraTransform.getT();
    auto tcami = m_cameraTransform.getInverse();
    auto re = -tcami.get3x3()*te;
    auto e1e = tcami.getX();
    e1e.normalize();
    auto e2e = tcami.getY();
    e2e.normalize();
    auto& rc = asVl(m_centerPosition);
    auto L = (re - rc).length();
    auto vfactor = L * m_options.panningSensitivity * make_real(tan(0.5*m_fovY*M_PI/180) / m_viewportSize[1]);
    auto dr = delta * vfactor;
    te[0] += dr[0];
    te[1] += dr[1];
    m_cameraTransform.setT(te);
    auto toVec3r = [](const vl::Vector3<s3dmm::real_type>& r) {
        return Vec3r {r[0], r[1], r[2]};
    };
    m_centerPosition -= toVec3r(e1e*dr[0] + e2e*dr[1]);
}

void CameraController::zoom(real_type delta)
{
    auto v = std::exp(-delta * m_options.zoomingSensitivity);
    auto Pt = m_cameraTransform.get3x3();
    auto P = Pt.getTransposed();
    auto re = -P * m_cameraTransform.getT();
    auto& rc = asVl(m_centerPosition);
    re = rc + v*(re - rc);
    m_cameraTransform.setT(-Pt * re);
}

void CameraController::zoom(const Vec2r& pos, real_type delta)
{
    // TODO better
    zoom(delta);
}

void CameraController::computeCurrentRotation()
{
    auto P = m_cameraTransform.get3x3().getTransposed();
    auto re = -P * m_cameraTransform.getT();
    auto& rc = asVl(m_centerPosition);
    auto e3e = re - rc;
    auto L_e3 = e3e.length();
    if (L_e3 <= 0) {
        e3e = vlVec3r(0, 0, 1);
        L_e3 = 1;
    }
    e3e /= L_e3;
    auto e1e = vl::cross(asVl(m_options.upDirection), e3e);
    auto L_e1 = e1e.length();
    if (L_e1 >= 0)
        e1e /= L_e1;
    else {
        auto e1e = vl::cross(otherDirVector(e3e), e3e);
        auto L_e1 = e1e.length();
        BOOST_ASSERT(L_e1 > 0);
        e1e /= L_e1;
    }
    auto e2e = vl::cross(e3e, e1e);
    m_cameraTransform.setX(vlVec3r(e1e[0], e2e[0], e3e[0]));
    m_cameraTransform.setY(vlVec3r(e1e[1], e2e[1], e3e[1]));
    m_cameraTransform.setZ(vlVec3r(e1e[2], e2e[2], e3e[2]));
}

} // namespace s3vs
