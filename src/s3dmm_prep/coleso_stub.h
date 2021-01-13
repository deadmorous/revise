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

enum tFuncType {
    FUNC_SCALAR          = 0, // scalar function
    FUNC_PULSATIONS      = 1, // physical pulsations (rho', u', v', w', p')
    FUNC_PHYSICAL        = 2, // physical variables (rho, u, v, w, p)
    FUNC_CONSERVATIVE    = 3, // conservative variables (rho, rho*u, rho*v, rho*w, E) or its pulsations
    FUNC_TEMPVEL         = 4, // velocities and temperature (1, u, v, w, p/rho)
    FUNC_PULSATIONS_COMPLEX = 5, // complex physical pulsations in cyl. coord. system ( Re(rho', uz', ur', uphi', p'), Im(...) )
    FUNC_PULSCONS_COMPLEX = 6  // complex conservative pulsations in cyl. coord. system
};

struct tFileBuffer;

struct tPointFunction {
    virtual const char* description() const = 0;
    virtual tFuncType Type() const = 0;
    inline int NumVars() const {
        switch(Type()) {
        case FUNC_SCALAR: return 1;
        case FUNC_TEMPVEL: return 5;
        case FUNC_PULSATIONS: case FUNC_PHYSICAL: case FUNC_CONSERVATIVE: return 5;
        case FUNC_PULSATIONS_COMPLEX: case FUNC_PULSCONS_COMPLEX: return 10;
        default: return 0;
        }
    }
    static constexpr int NumVarsMax = 10;

    virtual void SetEquationConstants(double /*_gam*/, double /*_Rey*/, double /*_Pr*/, 
        double /*_SoundVel*/, double /*_FlowVel*/) {}

    virtual void PointValue(double t, const double* coor, double* V) const = 0;
    virtual ~tPointFunction(void) {}

    virtual const char* filename() const {
        return nullptr;
    }
    virtual void ReadParams(struct tFileBuffer&) {}
    virtual void ReadParamsFromFile(const char* /*fname*/) {}
    virtual void Init() {}
};
