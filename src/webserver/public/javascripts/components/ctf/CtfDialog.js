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

    class CtfDialog extends objects.Dialog {
        constructor() {
            super("ctf-dialog");
            this.opacityInput = new ctf.OpacityInput();
            this.colormapInput = new ctf.ColormapInput();
            this._init();
        }
        setBackgroundColor(rgb) {
            this.opacityInput.setBackgroundColor(rgb);
            this.colormapInput.setBackgroundColor(rgb);
        }
        getCtf() {
            let opacityPoints = this.opacityInput
                .getPoints()
                .sort((op1, op2) => op1.relativeVal - op2.relativeVal);
            let colormapPoints = this.colormapInput
                .getPoints()
                .sort((cmPt1, cmPt2) => cmPt1.relativeVal - cmPt2.relativeVal);
            
            
            let ctf = new objects.Ctf.Func();
            ctf.colormapPoints = colormapPoints;
            ctf.opacityPoints = opacityPoints;
            return ctf;
        }
        setCtf(ctf) {
            this.opacityInput.attachColormapPoints(ctf.colormapPoints);
            this.colormapInput.setPoints(ctf.colormapPoints);
            this.opacityInput.setPoints(ctf.opacityPoints);
        }
        reset() {
            // TODO
        }

        // private
        _init() {
            let black = new objects.Rgb(0, 0, 0);

            this.opacityInput.setDefault();
            this.colormapInput.setDefault();

            this.opacityInput.setBackgroundColor(black);
            this.colormapInput.setBackgroundColor(black);
            
            let CTFunction = new ctf.Func();
            CTFunction.setDefault();
            this.setCtf(CTFunction);
            
            let self = this;
            this.colormapInput.addEventListener("change", e => {
                let colormapPoints = self.colormapInput.getPoints();
                self.opacityInput.attachColormapPoints(colormapPoints);
            });
        }
    }
    ctf.Dialog = CtfDialog
})()