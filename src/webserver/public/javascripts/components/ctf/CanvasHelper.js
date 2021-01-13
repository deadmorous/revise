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

    class CanvasHelper {
        constructor() {
            this.coordinatesTransform = null;
            this.painterOptions = null;
        }
        isPointerOnPoint(pointerX, pointerY, point) {
            let w = this.painterOptions.pointOptions.outRectSize;
            let h = w;
            let x = this.coordinatesTransform.toScreenX(point.x) - w / 2;
            let y = this.coordinatesTransform.toScreenY(point.y) - h / 2;
            let rect = new ctf.Rect(x, y, w, h);
            return rect.containsPoint(new ctf.Point(pointerX, pointerY));
        }
        fromScreenX(x) {
            return this.coordinatesTransform.fromScreenX(x);
        }
        fromScreenY(y) {
            return this.coordinatesTransform.fromScreenY(y);
        }
        toScreenY(y) {
            return this.coordinatesTransform.toScreenY(y);
        }
    }
    ctf.CanvasHelper = CanvasHelper;
})()