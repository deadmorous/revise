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

#include "s3vs/VsControllerInput.hpp"

namespace s3vs
{

std::vector<float> makeColorTransferVector(
    const ColorTransferFunction& ctmap,
    unsigned int colorCount)
{
    std::vector<float> ctf(colorCount*4);
    if (ctmap.empty()) {
        // Fill texture with white
        std::fill(ctf.begin(), ctf.end(), 1.f);
    }
    else if (ctmap.size() == 1) {
        // Fill texture with the only color in the map
        auto color = (ctmap.begin()->second).convertTo<float>();
        auto ctfc = reinterpret_cast<s3dmm::Vec<4, float>*>(ctf.data());
        std::fill(ctfc, ctfc+colorCount, color);
    }
    else {
        // Piecewise-linearly interpolate texture
        auto xmin = ctmap.begin()->first;
        auto xmax = ctmap.rbegin()->first;
        auto ctfc = reinterpret_cast<s3dmm::Vec<4, float>*>(ctf.data());
        auto it = ctmap.begin();
        auto c1 = it->second.convertTo<float>();
        ctfc[0] = c1;
        auto i1 = 0u;
        auto icolor = 1u;
        for (++it; it!=ctmap.end(); ++it, ++icolor) {
            auto x2 = it->first;
            auto c2 = it->second.convertTo<float>();
            auto i2 = icolor+1 == ctmap.size() ?
                                                 colorCount-1u:
                                                 static_cast<unsigned int>((x2-xmin)/(xmax-xmin)*colorCount);
            if (i2 > i1) {
                for (auto i=i1+1; i<=i2; ++i) {
                    auto t = static_cast<float>(i-i1)/(i2-i1);
                    ctfc[i] = c1*(1-t) + c2*t;
                }
            }
            c1 = c2;
            i1 = i2;
        }
    }
    return ctf;
}

std::vector<s3dmm::Vec<4, float>> makeColorTransferVectorVec4(
    const ColorTransferFunction& ctmap,
    unsigned int colorCount)
{
    auto ctf = makeColorTransferVector(ctmap, colorCount);
    std::vector<s3dmm::Vec<4, float>> res(colorCount);
    if (colorCount)
        memcpy(res.data(), ctf.data(), ctf.size()*sizeof(float));
    return res;
}


s3dmm::Vec<4, float> calcColor(const std::vector<s3dmm::Vec<4, float>>& ctv, float param)
{
    if (param <= 0.f || ctv.size() == 1u)
        return ctv.front();
    if (param >= 1.f)
        return ctv.back();
    auto par = param*(ctv.size() - 1);
    auto idx = static_cast<size_t>(par);
    auto t = par - idx;
    auto c = ctv[idx]*(1-t) + ctv[idx+1]*t;
    return c;
}

} // namespace s3vs
