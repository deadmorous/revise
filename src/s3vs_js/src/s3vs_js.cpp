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

#include <iostream>

#include "napi_helpers_s3vs.hpp"

#include <boost/dll/shared_library.hpp>
#include <boost/system/system_error.hpp>

using namespace s3vs;

SILVER_BULLETS_BEGIN_DEFINE_ENUM_NAMES(FieldMode)
    { FieldMode::Isosurface, "Isosurface"},
    { FieldMode::Isosurfaces, "Isosurfaces"},
    { FieldMode::MaxIntensityProjection, "MaxIntensityProjection"},
    { FieldMode::Argb, "Argb"},
    { FieldMode::ArgbLight, "ArgbLight"},
    { FieldMode::DomainVoxels, "DomainVoxels"},
    { FieldMode::ValueOnIsosurface, "ValueOnIsosurface"},
    { FieldMode::ValueOnIsosurfaces, "ValueOnIsosurfaces"}
SILVER_BULLETS_END_DEFINE_ENUM_NAMES();



class VsControllerFieldParamJs : public Napi::ObjectWrap<VsControllerFieldParamJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& /*exports*/)
    {
        Napi::Function func = DefineClass(env, "VsControllerFieldParam", {});

        // See https://github.com/nodejs/node-addon-api/blob/master/doc/object_wrap.md
        m_constructor = Napi::Persistent(func);
        m_constructor.SuppressDestruct();
        // exports.Set("VsControllerFieldParam", func);
    }

    explicit VsControllerFieldParamJs(const Napi::CallbackInfo& info) :
        Napi::ObjectWrap<VsControllerFieldParamJs>(info)
    {
    }

    static Napi::FunctionReference& constructor() {
        return m_constructor;
    }

    void setFieldParam(const SyncVsFieldAllParam& fieldParam)
    {
        if (m_fieldParam && fieldParam != m_fieldParam)
            throw std::runtime_error("VsControllerInputJs::setFieldParam() may only be called once");
        if (fieldParam != m_fieldParam) {
            m_fieldParam = fieldParam;
            auto thisObj = Value().As<Napi::Object>();
            thisObj.DefineProperty(makePropClassAccessor<WithThreshold>("threshold", m_fieldParam));
            thisObj.DefineProperty(makePropClassAccessor<WithColorTransferFunction>("colorTransferFunction", m_fieldParam));
            thisObj.DefineProperty(makePropClassAccessor<WithIsosurfaceLevel>("isosurfaceLevel", m_fieldParam));
            thisObj.DefineProperty(makePropClassAccessor<WithIsosurfaceLevels>("isosurfaceLevels", m_fieldParam));
            thisObj.DefineProperty(makePropClassAccessor<WithIsosurfaceOpacity>("isosurfaceOpacity", m_fieldParam));
            thisObj.DefineProperty(makePropClassAccessor<WithSecondaryField>("secondaryField", m_fieldParam));
        }
    }

    SyncVsFieldAllParam fieldParam(const Napi::CallbackInfo&)
    {
        return m_fieldParam;
    }

private:
    static Napi::FunctionReference m_constructor;
    SyncVsFieldAllParam m_fieldParam;
};

Napi::FunctionReference VsControllerFieldParamJs::m_constructor;



class ComputingCapsJs : public Napi::ObjectWrap<ComputingCapsJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& /*exports*/)
    {
        Napi::Function func = DefineClass(env, "ComputingCaps", {});

        // See https://github.com/nodejs/node-addon-api/blob/master/doc/object_wrap.md
        m_constructor = Napi::Persistent(func);
        m_constructor.SuppressDestruct();
        // exports.Set("ComputingCaps", func);
    }

    explicit ComputingCapsJs(const Napi::CallbackInfo& info) :
      Napi::ObjectWrap<ComputingCapsJs>(info)
    {
    }

    static Napi::FunctionReference& constructor() {
        return m_constructor;
    }

    void setComputingCaps(const SyncComputingCaps& computingCaps)
    {
        if (m_compuCaps && computingCaps != m_compuCaps)
            throw std::runtime_error("VsControllerInputJs::setComputingCaps() may only be called once");
        if (computingCaps != m_compuCaps) {
            m_compuCaps = computingCaps;
            auto thisObj = Value().As<Napi::Object>();
            thisObj.DefineProperty(makePropClassAccessor<WithCompNodeCount>("compNodeCount", m_compuCaps));
            thisObj.DefineProperty(makePropClassAccessor<WithGPUPerNodeCount>("GPUPerNodeCount", m_compuCaps));
            thisObj.DefineProperty(makePropClassAccessor<WithCPUPerNodeCount>("CPUPerNodeCount", m_compuCaps));
            thisObj.DefineProperty(makePropClassAccessor<WithWorkerThreadPerNodeCount>("workerThreadPerNodeCount", m_compuCaps));
        }
    }

    SyncComputingCaps computingCaps(const Napi::CallbackInfo&)
    {
        return m_compuCaps;
    }

private:
    static Napi::FunctionReference m_constructor;
    SyncComputingCaps m_compuCaps;
};

Napi::FunctionReference ComputingCapsJs::m_constructor;



class VsControllerOptsJs : public Napi::ObjectWrap<VsControllerOptsJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& /*exports*/)
    {
        Napi::Function func = DefineClass(env, "VsControllerOpts", {});

        // See https://github.com/nodejs/node-addon-api/blob/master/doc/object_wrap.md
        m_constructor = Napi::Persistent(func);
        m_constructor.SuppressDestruct();
        // exports.Set("VsControllerOpts", func);
    }

    explicit VsControllerOptsJs(const Napi::CallbackInfo& info) :
      Napi::ObjectWrap<VsControllerOptsJs>(info)
    {
    }

    static Napi::FunctionReference& constructor() {
        return m_constructor;
    }

    void setControllerOpts(const SyncVsControllerOpts& controllerOpts)
    {
        if (m_opts && controllerOpts != m_opts)
            throw std::runtime_error("VsControllerInputJs::setControllerOpts() may only be called once");
        if (controllerOpts != m_opts) {
            m_opts = controllerOpts;
            auto thisObj = Value().As<Napi::Object>();
            thisObj.DefineProperty(makePropClassAccessor<WithMeasureRenderingTime>("measureRenderingTime", m_opts));
        }
    }

    SyncVsControllerOpts controllerOpts(const Napi::CallbackInfo&)
    {
        return m_opts;
    }

private:
    static Napi::FunctionReference m_constructor;
    SyncVsControllerOpts m_opts;
};

Napi::FunctionReference VsControllerOptsJs::m_constructor;



class VsControllerInputJs : public Napi::ObjectWrap<VsControllerInputJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& /*exports*/)
    {
        Napi::Function func = DefineClass(env, "VsControllerInput", {});

        m_constructor = Napi::Persistent(func);
        m_constructor.SuppressDestruct();
        // exports.Set("VsControllerInput", func);
    }

    explicit VsControllerInputJs(const Napi::CallbackInfo& info) :
        Napi::ObjectWrap<VsControllerInputJs>(info)
    {
    }

    static Napi::FunctionReference& constructor() {
        return m_constructor;
    }

    void setInput(const SyncVsControllerInput& input)
    {
        if (m_input && input != m_input)
            throw std::runtime_error("VsControllerInputJs::setInput() may only be called once");
        if (input != m_input) {
            m_input = input;
            auto thisObj = Value().As<Napi::Object>();
            thisObj.DefineProperty(makePropClassAccessor<WithProblemPath>("problemPath", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithTimeFrame>("timeFrame", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithPrimaryField>("primaryField", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithFieldMode>("fieldMode", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithClippingPlanes>("clippingPlanes", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithViewportSize>("viewportSize", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithBackgroundColor>("backgroundColor", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithFovY>("fovY", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithRenderPatience>("renderPatience", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithRenderQuality>("renderQuality", m_input));
            thisObj.DefineProperty(makePropClassAccessor<WithRenderLevel>("renderLevel", m_input));

            thisObj.DefineProperty(Napi::PropertyDescriptor::Accessor<&VsControllerInputJs::fieldParam>(
                        "fieldParam", napi_enumerable, &m_fieldParam));

            m_fieldParam = Napi::Persistent(VsControllerFieldParamJs::constructor().New(std::vector<napi_value>{}));
            VsControllerFieldParamJs::Unwrap(m_fieldParam.Value())->setFieldParam(
                {m_input.unsafeData()->fieldAllParam(), *m_input.lockable()});
        }

        // deBUG, TODO: Remove
//        m_input->fieldAllParam().onThresholdChanged([]() {
//            std::cout << "FIELD THRESHOLD HAS CHANGED" << std::endl;
//        });

    }

    SyncVsControllerInput input() const {
        return m_input;
    }

    static Napi::Value fieldParam(const Napi::CallbackInfo& info)
    {
        return reinterpret_cast<Napi::ObjectReference*>(info.Data())->Value();
    }

private:
    static Napi::FunctionReference m_constructor;
    SyncVsControllerInput m_input;
    Napi::ObjectReference m_fieldParam;
};

Napi::FunctionReference VsControllerInputJs::m_constructor;

Napi::ObjectReference m_computingCaps;
Napi::ObjectReference m_controllerOpts;

class VsControllerJs : public Napi::ObjectWrap<VsControllerJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& exports)
    {
        Napi::Function constructor = DefineClass(
            env, "VsController", {
                InstanceAccessor<&VsControllerJs::computingCaps>("computingCaps"),
                InstanceAccessor<&VsControllerJs::controllerOpts>("controllerOpts"),
                InstanceMethod<&VsControllerJs::start>("start"),
                InstanceAccessor<&VsControllerJs::input>("input"),
                InstanceAccessor<&VsControllerJs::cameraTransform, &VsControllerJs::setCameraTransform>("cameraTransform"),
                InstanceAccessor<&VsControllerJs::cameraCenterPosition, &VsControllerJs::setCameraCenterPosition>("cameraCenterPosition"),
                InstanceMethod<&VsControllerJs::resetCameraTransform>("resetCameraTransform"),
                InstanceAccessor<&VsControllerJs::fieldNames>("fieldNames"),
                InstanceMethod<&VsControllerJs::fieldRange>("fieldRange"),
                InstanceMethod<&VsControllerJs::timeValues>("timeValues"),
                InstanceAccessor<&VsControllerJs::timeFrameCount>("timeFrameCount"),
                InstanceMethod<&VsControllerJs::timeByTimeFrame>("timeByTimeFrame"),
                InstanceMethod<&VsControllerJs::timeFrameByTime>("timeFrameByTime"),
                InstanceMethod<&VsControllerJs::updateMouseState>("updateMouseState"),
                InstanceAccessor<&VsControllerJs::frameOutput>("frameOutput"),
                InstanceMethod<&VsControllerJs::kill>("kill")
            });
        exports.Set("VsController", constructor);
    }

    explicit VsControllerJs(const Napi::CallbackInfo& info) :
        Napi::ObjectWrap<VsControllerJs>(info)
    {
        auto loadLib = [&info](boost::dll::shared_library& lib, const char *name)
        {
            if (!lib) {
                if (m_s3vsBinaryDir.empty()) {
                    if (info.Length() < 1)
                        throw Napi::Error::New(info.Env(), "An argument was expected (path to s3vs binary directory)");
                    m_s3vsBinaryDir = static_cast<std::string>(info[0].ToString());
                }
                boost::system::error_code ec;
                lib.load(m_s3vsBinaryDir + "/" + name, ec);
                if (ec)
                    throw boost::system::system_error(ec);
            }
        };
        loadLib(m_s3vs_lib, "libs3vs_lib.so");
        loadLib(m_s3vs_worker, "libs3vs_worker.so");
        m_vsctl = VsControllerInterface::newInstance("default");
        m_vsctl->setShaderPath(m_s3vsBinaryDir + "/data");

        m_input = Napi::Persistent(VsControllerInputJs::constructor().New(std::vector<napi_value>{}));
        VsControllerInputJs::Unwrap(m_input.Value())->setInput(m_vsctl->input());

        m_computingCaps = Napi::Persistent(ComputingCapsJs::constructor().New(std::vector<napi_value>{}));
        ComputingCapsJs::Unwrap(m_computingCaps.Value())->setComputingCaps(m_vsctl->computingCaps());

        m_controllerOpts = Napi::Persistent(VsControllerOptsJs::constructor().New(std::vector<napi_value>{}));
        VsControllerOptsJs::Unwrap(m_controllerOpts.Value())->setControllerOpts(m_vsctl->controllerOpts());
    }

    Napi::Value computingCaps(const Napi::CallbackInfo&)
    {
        return m_computingCaps.Value();
    }

    Napi::Value controllerOpts(const Napi::CallbackInfo&)
    {
        return m_controllerOpts.Value();
    }

    Napi::Value start(const Napi::CallbackInfo& info)
    {
        m_vsctl->start();
        return Napi::Value();
    }

    Napi::Value input(const Napi::CallbackInfo&)
    {
        return m_input.Value();
    }

    void setCameraTransform(const Napi::CallbackInfo& info, const Napi::Value& cameraTransform)
    {
        Matrix4r m;
        readFixedLengthArrayObj<16>(m.ptr(), cameraTransform);
        m_vsctl->setCameraTransform(m);
    }

    Napi::Value cameraTransform(const Napi::CallbackInfo& info)
    {
        auto m = m_vsctl->cameraTransform();
        return makeArrayObj(info.Env(), m.ptr(), 16);
    }

    Napi::Value resetCameraTransform(const Napi::CallbackInfo& info)
    {
        m_vsctl->resetCameraTransform();
        return Napi::Value();
    }

    Napi::Value cameraCenterPosition(const Napi::CallbackInfo& info) {
        Vec3r c = m_vsctl->cameraCenterPosition();
        return makeArrayObj(info.Env(), c.data(), 3);
    }

    void setCameraCenterPosition(const Napi::CallbackInfo& info, const Napi::Value& centerPosition) {
        Vec3r c;
        readFixedLengthArrayObj<3>(c.data(), centerPosition);
        m_vsctl->setCameraCenterPosition(c);
    }

    Napi::Value fieldNames(const Napi::CallbackInfo& info) {
        return makeArrayObj(info.Env(), m_vsctl->fieldNames());
    }

    Napi::Value fieldRange(const Napi::CallbackInfo& info)
    {
        if (info.Length() != 1)
            throw Napi::Error::New(info.Env(), "An argument was expected (field name)");
        auto fieldNameArg = info[0];
        if (!fieldNameArg.IsString())
            throw Napi::Error::New(info.Env(), "Field name argument must be a string");
        try {
            return makeArrayObj(info.Env(), m_vsctl->fieldRange(fieldNameArg.As<Napi::String>()));
        }
        catch(std::exception& e) {
            throw Napi::Error::New(info.Env(), e.what());
        }
    }

    Napi::Value timeValues(const Napi::CallbackInfo& info)
    {
        return makeArrayObj(info.Env(), m_vsctl->timeValues());
    }

    Napi::Value timeFrameCount(const Napi::CallbackInfo& info)
    {
        return Napi::Number::New(info.Env(), m_vsctl->timeFrameCount());
    }

    Napi::Value timeByTimeFrame(const Napi::CallbackInfo& info)
    {
        if (info.Length() != 1)
            throw Napi::Error::New(info.Env(), "An argument was expected (time frame)");
        auto timeFrameArg = info[0];
        if (!timeFrameArg.IsNumber())
            throw Napi::Error::New(info.Env(), "Time frame argument must be a number");
        auto timeFrame = static_cast<unsigned int>(timeFrameArg.As<Napi::Number>());
        return Napi::Number::New(info.Env(), m_vsctl->timeByTimeFrame(timeFrame));
    }

    Napi::Value timeFrameByTime(const Napi::CallbackInfo& info)
    {
        if (info.Length() != 1)
            throw Napi::Error::New(info.Env(), "An argument was expected (time value)");
        auto timeArg = info[0];
        if (!timeArg.IsNumber())
            throw Napi::Error::New(info.Env(), "Time argument must be a number");
        auto time = static_cast<s3dmm::real_type>(timeArg.As<Napi::Number>());
        return Napi::Number::New(info.Env(), m_vsctl->timeFrameByTime(time));
    }

    void updateMouseState(const Napi::CallbackInfo& info)
    {
        if (info.Length() != 1)
            throw Napi::Error::New(info.Env(), "An argument was expected (mouse state)");
        auto mouseStateArg = info[0];
        if (!mouseStateArg.IsObject())
            throw Napi::Error::New(info.Env(), "Mouse state argument must be an object");
        auto mouseState = mouseStateArg.As<Napi::Object>();
        MouseState ms;
        ms.x = mouseState.Get("x").ToNumber();
        ms.y = mouseState.Get("y").ToNumber();
        ms.wheelDelta = mouseState.Get("wheelDelta").ToNumber();
        ms.flags = mouseState.Get("flags").ToNumber();
        m_vsctl->updateMouseState(ms);
    }

    Napi::Value frameOutput(const Napi::CallbackInfo& info)
    {
        auto result = Napi::Object::New(info.Env());
        auto fo = m_vsctl->frameOutput();
        result["shmem"] = fo.shmem->shmid();
        result["frameSize"] = fo.frameSize;
        result["frameWidth"] = fo.frameWidth;
        result["frameHeight"] = fo.frameHeight;
        return result;
    }

    void kill(const Napi::CallbackInfo&) {
        m_vsctl->kill();
    }

private:
    std::shared_ptr<VsControllerInterface> m_vsctl;
    Napi::ObjectReference m_computingCaps;
    Napi::ObjectReference m_controllerOpts;
    Napi::ObjectReference m_input;
    static std::string m_s3vsBinaryDir;
    static boost::dll::shared_library m_s3vs_lib;
    static boost::dll::shared_library m_s3vs_worker;
};

std::string VsControllerJs::m_s3vsBinaryDir;
boost::dll::shared_library VsControllerJs::m_s3vs_lib;
boost::dll::shared_library VsControllerJs::m_s3vs_worker;

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    VsControllerFieldParamJs::Init(env, exports);
    ComputingCapsJs::Init(env, exports);
    VsControllerOptsJs::Init(env, exports);
    VsControllerInputJs::Init(env, exports);
    VsControllerJs::Init(env, exports);
    return exports;
}

NODE_API_MODULE(s3vs_js, Init)
