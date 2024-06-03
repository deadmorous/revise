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

#include "s3vs/VsControllerInterface.hpp"
#include "s3vs/VsWorkerInterface.hpp"

#include "s3dmm/BlockTreeFieldService.hpp"

#include "silver_bullets/system/get_program_dir.hpp"

#include "VsRenderer.hpp"
#include "MouseInterpreter.hpp"
#include "CameraController.hpp"

#include <boost/assert.hpp>

#include <mutex>
#include <thread>

using namespace std;
using namespace s3dmm;

namespace s3vs
{

class VsController :
        public VsControllerInterface,
        public silver_bullets::FactoryMixin<VsController, VsControllerInterface>,
        boost::noncopyable
{
public:
    VsController() :
        m_renderer(make_shared<VsRenderer>(&m_input, m_compuCaps, m_opts)),
        m_mouseInterpreter(
            { m_input, m_inputMutex },
            m_cameraController, [this](const Matrix4r& cameraTransform) {
                setRendererCameraTransform(cameraTransform);
            })
    {
        m_input.onProblemPathChanged([this]() {
            m_cameraController.resetCameraTransform();
            m_renderer->onProblemPathChanged();
        });
        m_input.onTimeFrameChanged([this]() { m_renderer->onTimeFrameChanged(); });
        m_input.onPrimaryFieldChanged([this]() { m_renderer->onPrimaryFieldChanged(); });
        m_input.onFieldModeChanged([this]() { m_renderer->onFieldModeChanged(); });
        m_input.onClippingPlanesChanged([this]() { m_renderer->onClippingPlanesChanged(); });
        m_input.onViewportSizeChanged([this]() {
            m_renderer->onViewportSizeChanged();
            m_cameraController.setViewportSize(m_input.viewportSize().convertTo<unsigned int>());
        });
        m_input.onBackgroundColorChanged([this]() { m_renderer->onBackgroundColorChanged(); });
        m_input.onFovYChanged([this]() {
            m_renderer->onFovYChanged();
            m_cameraController.setFovY(m_input.fovY());
        });
        m_input.onRenderPatienceChanged([this]() { m_renderer->onRenderPatienceChanged(); });
        m_input.onRenderQualityChanged([this]() { m_renderer->onRenderQualityChanged(); });
        m_input.onRenderLevelChanged([this]() { m_renderer->onRenderLevelChanged(); });

        auto& fieldParam = m_input.fieldAllParam();
        fieldParam.onThresholdChanged([this]() { m_renderer->onFieldThresholdChanged(); });
        fieldParam.onColorTransferFunctionChanged([this]() { m_renderer->onFieldColorTransferFunctionChanged(); });
        fieldParam.onIsosurfaceLevelChanged([this]() { m_renderer->onFieldIsosurfaceLevelChanged(); });
        fieldParam.onIsosurfaceLevelsChanged([this]() { m_renderer->onFieldIsosurfaceLevelsChanged(); });
        fieldParam.onIsosurfaceOpacityChanged([this]() { m_renderer->onFieldIsosurfaceOpacityChanged(); });
        fieldParam.onSecondaryFieldChanged([this]() { m_renderer->onFieldSecondaryFieldChanged(); });
    }

    ~VsController()
    {
        cout << "~~~~~~~~~~~~~ ~VsController() ..." << endl;    // deBUG, TODO: Remove
        kill();
    }

    void setShaderPath(const string& shaderPath) override {
        m_renderer->setShaderPath(shaderPath);
    }

    string shaderPath() const override {
        return m_renderer->shaderPath();
    }

    SyncComputingCaps computingCaps() override {
        return { m_compuCaps, m_inputMutex };
    }

    SyncVsControllerOpts controllerOpts() override {
        return { m_opts, m_inputMutex };
    }

    void start() override {
        m_renderer->start();
    }

    SyncVsControllerInput input() override {
        return { m_input, m_inputMutex };
    }

    void setCameraTransform(const Matrix4r& cameraTransform) override
    {
        m_cameraController.setCameraTransform(cameraTransform);
        setRendererCameraTransform(cameraTransform);
    }

    Matrix4r cameraTransform() const override {
        return m_renderer->cameraTransform();
    }

    void resetCameraTransform() override {
        m_cameraController.resetCameraTransform();
    }

    void setCameraCenterPosition(const Vec3r& centerPosition) override {
        m_cameraController.setCenterPosition(centerPosition);
    }

    Vec3r cameraCenterPosition() const override {
        return m_cameraController.centerPosition();
    }

    vector<string> fieldNames() const override {
        return m_renderer->fieldNames();
    }

    Vec2r fieldRange(const std::string& fieldName) const override {
        return m_renderer->fieldRange(fieldName);
    }

    std::vector<s3dmm::real_type> timeValues() const override {
        unsigned int nTimeFrames = timeFrameCount();
        std::vector<s3dmm::real_type> timeValues(nTimeFrames);
        for(unsigned int i = 0; i < nTimeFrames; ++i)
            timeValues[i] = static_cast<real_type>(timeByTimeFrame(i));
        return timeValues;
    }

    unsigned int timeFrameCount() const override {
        return m_renderer->timeFrameCount();
    }

    real_type timeByTimeFrame(unsigned int timeFrame) const override {
        return static_cast<real_type>(timeFrame);   // TODO
    }

    unsigned int timeFrameByTime(real_type time) const override {
        return static_cast<unsigned int>(time);   // TODO
    }

    void updateMouseState(const MouseState& mouseState) override {
        m_mouseInterpreter.interpretMouse(mouseState);
    }

    VsControllerFrameOutput frameOutput() const override
    {
        lock_guard<mutex> g(m_inputMutex);
        return m_renderer->frameOutput();
    }

    void kill() override
    {
        cout << "~~~~~~~~~~~~~ VsController::kill() ..." << endl;    // deBUG, TODO: Remove
        m_renderer.reset();
    }

private:
    ComputingCaps m_compuCaps;
    VsControllerOpts m_opts;
    VsControllerInput m_input;
    mutable mutex m_inputMutex;
    shared_ptr<VsRenderer> m_renderer;
    CameraController m_cameraController;
    MouseInterpreter m_mouseInterpreter;

    void setRendererCameraTransform(const Matrix4r& cameraTransform) {
        m_renderer->setCameraTransform(cameraTransform);
    }
};

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VsController, "default");

} // namespace s3vs
