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

#include "ComputingCaps.hpp"
#include "VsControllerInput.hpp"
#include "MouseState.hpp"
#include "VsControllerFrameOutput.hpp"
#include "VsControllerOpts.hpp"

#include "silver_bullets/factory.hpp"

namespace s3vs
{

/**
 * @brief Visualization server controller interface
 *
 * Methods provide functionality to control the behavior of the visualization server.
 */
struct VsControllerInterface : public silver_bullets::Factory<VsControllerInterface>
{
    virtual ~VsControllerInterface() = default;

    /**
     * @brief Sets path to the directory where shaders can be found.
     * @param shaderPath Path to directory containing shaders used by render server.
     *
     * This method has to be called before setting any value on object returned by
     * the #input() method.
     * @note By default, shader path is program_dir/data, where program_dir is
     * the directory where the process executable file is located.
     */
    virtual void setShaderPath(const std::string& shaderPath) = 0;

    /**
     * @brief Returns current path to the directory where shaders can be found.
     * @sa #setShaderPath().
     */
    virtual std::string shaderPath() const = 0;

    /**
     * @brief Returns reference to the instance with the hardware computing capabilities.
     *
     * All changes in returned object must be done before call of #start() method,
     * when the rendering is not started yet.
     */
    virtual SyncComputingCaps computingCaps() = 0;

    /**
     * @brief Returns reference to the instance with the visualization controller options.
     *
     * All changes in returned object must be done before call of #start() method,
     * when the rendering is not started yet.
     */
    virtual SyncVsControllerOpts controllerOpts() = 0;

    /**
     * @brief Starts the rendering.
     */
    virtual void start() = 0;

    /**
     * @brief Returns reference to the instance containing controller input parameters.
     */
    virtual SyncVsControllerInput input() = 0;

    /**
     * @brief Sets camera transformation.
     * @param cameraTransform Transformation matrix to set as the camera tramsformation.
     */
    virtual void setCameraTransform(const Matrix4r& cameraTransform) = 0;

    /**
     * @brief Returns transformation matrix currently used for the camera.
     */
    virtual Matrix4r cameraTransform() const = 0;

    /**
     * @brief Resets camera transformation to its initial state.
     */
    virtual void resetCameraTransform() = 0;

    /**
     * @brief Sets camera center position
     * @param Camera center position
     */
    virtual void setCameraCenterPosition(const Vec3r& centerPosition) = 0;

    /**
     * @brief Returns camera center position
     */
    virtual Vec3r cameraCenterPosition() const = 0;

    /**
     * @brief Returns names of all available fields.
     *
     * In order this method to return something, it is necessary to specify
     * problem path first (see \ref WithProblemPath).
     */
    virtual std::vector<std::string> fieldNames() const = 0;

    /**
     * @brief Returns the total range of the specified field.
     * @param fieldName The name of the field - see \ref fieldNames().
     * @return Vector of two elements, minimum and maximum field values.
     */
    virtual Vec2r fieldRange(const std::string& fieldName) const = 0;

    /**
     * @brief Returns the array of time values.
     *
     * In order this method to return something, it is necessary to specify
     * problem path first (see \ref WithProblemPath).
     */
    virtual std::vector<s3dmm::real_type> timeValues() const = 0;

    /**
     * @brief Returns total number of time frames.
     *
     * In order this method to return something, it is necessary to specify
     * problem path first (see \ref WithProblemPath).
     */
    virtual unsigned int timeFrameCount() const = 0;

    /**
     * @brief Returns time value corresponding to the specified time frame.
     * @param timeFrame Number of the time frame, from 0 to timeFrameCount()-1.
     * @return Time corresponding to \a timeFrame.
     *
     * In order this method to return something, it is necessary to specify
     * problem path first (see \ref WithProblemPath).
     */
    virtual s3dmm::real_type timeByTimeFrame(unsigned int timeFrame) const = 0;

    /**
     * @brief Returns the number of time frame having the nearest time value.
     * @param time The value of time.
     * @return Number of the time frame, from 0 to timeFrameCount()-1, such that
     * its time is closer to \a time than the time of any other time frame.
     */
    virtual unsigned int timeFrameByTime(s3dmm::real_type time) const = 0;

    /**
     * @brief Updates camera position according to the current mouse state.
     * @param mouseState Client's mouse state data.
     */
    virtual void updateMouseState(const MouseState& mouseState) = 0;

    /**
     * @brief Returns format and location of output frames generated by the visualization server.
     */
    virtual VsControllerFrameOutput frameOutput() const = 0;

    /**
     * @brief Kills this instance of render server.
     *
     * The render server can no longer be used after this method is called.
     */
    virtual void kill() = 0;
};

} // namespace s3vs
