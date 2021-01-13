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

#include "napi_helpers.hpp"
#include "ws_sendframe/FrameServerThread.hpp"
#include "ws_sendframe/FrameServer.hpp"

class FrameServerJs : public Napi::ObjectWrap<FrameServerJs>
{
public:
    static void Init(Napi::Env& env, Napi::Object& exports)
    {
        Napi::Function constructor = DefineClass(env, "FrameServer", {
            InstanceMethod<&FrameServerJs::setSource>("setSource"),
            InstanceMethod<&FrameServerJs::removeSource>("removeSource"),
        });
        exports.Set("FrameServer", constructor);
    }

    explicit FrameServerJs(const Napi::CallbackInfo& info) :
        Napi::ObjectWrap<FrameServerJs>(info),
        m_frameServerThread(static_cast<unsigned short>(checkedArg<std::uint32_t>(info, 0)))
    {
    }

    Napi::Value setSource(const Napi::CallbackInfo& info)
    {
        if (info.Length() != 3)
            throw Napi::Error::New(info.Env(), "3 arguments were expected");
        auto sourceId = checkedArg<std::string>(info, 0);
        auto shmid = checkedArg<int>(info, 1);
        auto frameSize = toTInitializer<s3dmm::Vec2u>(info[2].As<Napi::Array>());
        m_frameServerThread.frameServer()->setSourceInfo(sourceId, shmid, frameSize);
        return Napi::Value();
    }

    Napi::Value removeSource(const Napi::CallbackInfo& info)
    {
        auto sourceId = checkedArg<std::string>(info, 0);
        m_frameServerThread.frameServer()->removeSource(sourceId);
        return Napi::Value();
    }

private:
    FrameServerThread m_frameServerThread;

    template<class T>
    static T checkedArg(const Napi::CallbackInfo& info, std::size_t index)
    {
        if (info.Length() < index)
            throw Napi::Error::New(
                    info.Env(),
                    std::string("Argument ") + boost::lexical_cast<std::string>(index+1) + " was expected");
        return Ntype<T>::toTInitializer(Ntype<T>::coerce(info[index]));
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    FrameServerJs::Init(env, exports);
    return exports;
}

NODE_API_MODULE(s3vs_js, Init)
