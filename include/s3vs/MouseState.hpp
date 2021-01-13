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

namespace s3vs
{

/// @brief The state of the mouse and some special keys.
struct MouseState
{
    /// @brief Flags determining the state of mouse buttons and special keyboard keys.
    enum Flags {
        /// @brief Left mouse button is held down.
        LeftButton      = 0x01,

        /// @brief Right mouse button is held down.
        RightButton     = 0x02,

        /// @brief Middle mouse button is held down.
        MiddleButton    = 0x04,

        /// @brief The Shift keyboard key is held down.
        ShiftKey        = 0x08,

        /// @brief The Ctrl keyboard key is held down.
        CtrlKey         = 0x10,

        /// @brief The Alt keyboard key is held down.
        AltKey          = 0x20
    };

    /// @brief Mouse pointer X position, relatively to the viewport (zero X is at the left boundary).
    int x = 0;

    /// @brief Mouse pointer Y position, relatively to the viewport (zero Y is at the top boundary).
    int y = 0;

    /// @brief Mouse wheel delta (usually one wheel tick equals 120), if user just rotated the wheel.
    int wheelDelta = 0;

    /// @brief A combination of #Flags enumeration values.
    unsigned int flags = 0;
};

} // namespace s3vs
