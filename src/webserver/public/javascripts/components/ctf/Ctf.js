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

(function(){
    let ctf = objects.Ctf;

    class ColormapPoint {
        constructor(relVal, rgb) {
            this.relativeVal = relVal;
            this.rgb = rgb;
        }
    }
    ctf.ColormapPoint = ColormapPoint;


    class OpacityPoint {
        constructor(relVal, opacity) {
            this.relativeVal = relVal;
            this.opacity = opacity;
        }
    }
    ctf.OpacityPoint = OpacityPoint;


    class Ctf {
        constructor() {
            this.opacityPoints = [];
            this.colormapPoints = [];
        }
        makeDataToStore() {
            let data = {
                opacityPoints: this.opacityPoints,
                colormapPoints: this.colormapPoints
            };
            return data;
        }
        makeDataToSend() {
            let interpolator = new ctf.Interpolate(this.opacityPoints, this.colormapPoints);
            this.opacityPoints = interpolator.opacityPoints;
            this.colormapPoints = interpolator.colormapPoints;
            let data = {};
            let id1 = 0;
            let id2 = 0;
            while(id1 != this.colormapPoints.length || id2 != this.opacityPoints.length) {
                let relVal1 = this.colormapPoints[id1].relativeVal;
                let relVal2 = this.opacityPoints[id2].relativeVal;
                if(relVal1 < relVal2) {
                    let rgb = this.colormapPoints[id1].rgb;
                    let opacity = interpolator.interpolateOpacity(relVal1);
                    data[relVal1] = [   rgb.r / 255,
                                        rgb.g / 255,
                                        rgb.b / 255,
                                        opacity
                                    ];
                    id1++;
                } else if(relVal1 > relVal2) {
                    let rgb = interpolator.interpolateColor(relVal2);
                    let opacity = this.opacityPoints[id2].opacity;
                    data[relVal2] = [   rgb.r / 255,
                                        rgb.g / 255,
                                        rgb.b / 255,
                                        opacity
                                    ];
                    id2++;
                } else {
                    let rgb = interpolator.colormapPoints[id1].rgb;
                    let opacity = this.opacityPoints[id2].opacity;
                    data[relVal2] = [   rgb.r / 255,
                                        rgb.g / 255,
                                        rgb.b / 255,
                                        opacity
                                    ];
                    id1++;
                    id2++;
                }
            }
            return data;
        }
        static restore(data) {
            let ctf = new Ctf();
            ctf.opacityPoints = data.opacityPoints
                .map(op => new OpacityPoint(op.relativeVal, op.opacity));
            ctf.colormapPoints = data.colormapPoints
                .map(cmPt => new ColormapPoint(
                    cmPt.relativeVal,
                    objects.Rgb.makeFromTuple(cmPt.rgb)
                ));
            return ctf;
        }
        setDefault() {
            this.opacityPoints = [];
            this.opacityPoints.push(new OpacityPoint(0, 1));
            this.opacityPoints.push(new OpacityPoint(1, 1));

            this.colormapPoints = [];
            this.colormapPoints.push(new ColormapPoint(0, new objects.Rgb(0, 0, 255)));
            this.colormapPoints.push(new ColormapPoint(0.25, new objects.Rgb(0, 255, 255)));
            this.colormapPoints.push(new ColormapPoint(0.5, new objects.Rgb(0, 255, 0)));
            this.colormapPoints.push(new ColormapPoint(0.75, new objects.Rgb(255, 255, 0)));
            this.colormapPoints.push(new ColormapPoint(1, new objects.Rgb(255, 0, 0)));
        }
    }
    objects.Ctf.Func = Ctf;


    class CtfInterpolate {
        constructor(opacityPoints, colormapPoints) {
            this.opacityPoints = null;
            this.colormapPoints = null;
            this.attachOpacityPoints(opacityPoints);
            this.attachColormapPoints(colormapPoints);
        }
        attachOpacityPoints(opacityPoints) {
            if(opacityPoints)
                this.opacityPoints = opacityPoints.sort((op1, op2) => op1.relativeVal - op2.relativeVal);
        }
        attachColormapPoints(colormapPoints) {
            if(colormapPoints)
                this.colormapPoints = colormapPoints.sort((cp1, cp2) => cp1.relativeVal - cp2.relativeVal);
        }
        interpolateOpacity(relVal) {
            if(!this.opacityPoints)
                return;
            if(this.opacityPoints.length == 1)
                return this.opacityPoints[0].opacity;
            
            if(relVal <= this.opacityPoints[0].relativeVal)
                return this.opacityPoints[0].opacity;
            else if(relVal >= this.opacityPoints[this.opacityPoints.length - 1].relativeVal)
                return this.opacityPoints[1].opacity;
            
            let idx = 0;    
            for(; this.opacityPoints[idx + 1].relativeVal < relVal; ++idx) {}
            let op1 = this.opacityPoints[idx].opacity;
            let op2 = this.opacityPoints[idx + 1].opacity;
            let relVal1 = this.opacityPoints[idx].relativeVal;
            let relVal2 = this.opacityPoints[idx + 1].relativeVal;
            return op1 + (op2 - op1) / (relVal2 - relVal1) * (relVal - relVal1);
        }
        interpolateColor(relVal) {
            if(this.colormapPoints.length == 1)
                return this.colormapPoints[0].rgb;

            if(relVal <= this.colormapPoints[0].relativeVal)
                return this.colormapPoints[0].rgb;
            else if(relVal >= this.colormapPoints[this.colormapPoints.length - 1].relativeVal)
                return this.colormapPoints[this.colormapPoints.length - 1].rgb;
            let idx = 0;
            for(; this.colormapPoints[idx + 1].relativeVal < relVal; ++idx) {}
            let rgb1 = this.colormapPoints[idx].rgb;
            let rgb2 = this.colormapPoints[idx + 1].rgb;
            let relVal1 = this.colormapPoints[idx].relativeVal;
            let relVal2 = this.colormapPoints[idx + 1].relativeVal;
            let r = rgb1.r + (rgb2.r - rgb1.r) * (relVal - relVal1) / (relVal2 - relVal1);
            let g = rgb1.g + (rgb2.g - rgb1.g) * (relVal - relVal1) / (relVal2 - relVal1);
            let b = rgb1.b + (rgb2.b - rgb1.b) * (relVal - relVal1) / (relVal2 - relVal1);
            return new objects.Rgb(r, g, b);
        }
    }
    ctf.Interpolate = CtfInterpolate;
})()