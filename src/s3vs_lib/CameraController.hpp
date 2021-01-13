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

#include "s3vs/types.hpp"

namespace s3vs
{

class CameraController
{
public:
    struct Options
    {
        Vec3r upDirection = { 0, 1, 0 };
        bool respectUpDirection = false;
        s3dmm::real_type rotationSensitivity = s3dmm::make_real(1);
        s3dmm::real_type panningSensitivity = s3dmm::make_real(1);
        s3dmm::real_type zoomingSensitivity = s3dmm::make_real(1);
    };

    CameraController();

    void resetCameraTransform();

    void setOptions(const Options& options);
    Options options() const;

    void setViewportSize(const s3dmm::Vec2u& viewportSize);
    void setCameraTransform(const Matrix4r& cameraTransform);
    void setCenterPosition(const Vec3r& centerPosition);
    void setFovY(s3dmm::real_type fovY);

    s3dmm::Vec2u viewportSize() const;
    Matrix4r cameraTransform() const;
    Vec3r centerPosition() const;
    s3dmm::real_type fovY() const;

    void rotate(const Vec2r& delta);
    void rotateInPlane(const Vec2r& delta);
    void pan(const Vec2r& delta);
    void zoom(s3dmm::real_type delta);
    void zoom(const Vec2r& pos, s3dmm::real_type delta);

    void computeCurrentRotation();
private:
    Options m_options;

    s3dmm::Vec2u m_viewportSize = { 320, 200 };
    Matrix4r m_cameraTransform;
    Vec3r m_centerPosition = { 0, 0, 0 };
    s3dmm::real_type m_fovY = 30;
};

} // namespace s3vs
