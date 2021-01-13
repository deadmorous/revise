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

#include <cstdint>

namespace s3dmm {

struct RenderKernelViewportParam
{
    std::uint32_t *pixels;      // Viewport pixels, row-wise, in the ARGB format, top rows first
    unsigned int W;             // Viewport width, in pixels
    unsigned int H;             // Viewport height, in pixels
};

struct RenderKernelViewParam
{
    unsigned int x;             // View origin abscissa, in pixels
    unsigned int y;             // View origin ordinate, in pixels
    unsigned int W;             // View width, in pixels
    unsigned int H;             // View height, in pixels
};

struct RenderKernelFieldParam
{
    cudaTextureObject_t tex;    // Texture object containing the field
    unsigned int texDepth;      // Texture depth: number of texels in each direction (is 1 + 2^texDepth)
    float minValue;             // Minimum value of the field
    float maxValue;             // Maximum value of the field
};

struct RenderKernelCubeParam
{
    float3 center;              // Cube center position, in world CS
    float halfSize;             // Cube half size, in world CS
    unsigned int level;         // Cube level in, Metadata sense
};

struct RenderKernelEyeParam
{
    float3 pos;                 // Eye position, in world CS
    float3 e1;                  // Unit vector of screen X direction, in world CS
    float3 e2;                  // Unit vector of screen Y direction, in world CS
    float3 n;                   // Vector of camera direction (in world CS), length = distance between eye and screen
    float w;                    // Screen width, in World CS
    float h;                    // Screen height, in World CS
    float pixelSize;            // Size of screen pixel, in world CS
};

struct LightSource
{
    float3 pos;
    float3 ambient;
    float3 diffuse;
    float3 specular;
};

struct RenderKernelPresentationParam
{
    float threshold;                // Field threshold or isosurface level value
    float isosurfaceOpacity;        // Isosurface opacity, multiplied by alpha values from color transfer function
    cudaTextureObject_t colorTex;   // Texture representing color transfer function
    float quality;                  // Sampling quality
    static constexpr unsigned int MaxLevels = 50;
    float levels[MaxLevels];        // Isosurface level values (for multi-level field modes)
    unsigned int levelCount;        // Isosurface level count
};

struct RenderKernelLightsParam
{
    static constexpr auto MaxSources = 10;
    LightSource sources[MaxSources];
    unsigned int sourceCount = 0;
};

struct RenderKernelParam
{
    // Viewport data
    RenderKernelViewportParam viewport;

    // View rectangle data
    RenderKernelViewParam view;

    // Field data
    RenderKernelFieldParam primaryField;
    RenderKernelFieldParam secondaryField;

    // Cube size and position
    RenderKernelCubeParam cube;

    // Eye data
    RenderKernelEyeParam eye;

    // Presentation parameters
    RenderKernelPresentationParam presentation;

    // Light sources
    RenderKernelLightsParam lights;
};

} // s3dmm
