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

    class Scene extends EventTarget {
        constructor() {
            super();
            this.painter = null;
            this.canvas = null;
            this.helper = null
            this.points = [];
            this.selectedPointIdx = -1;
        }
        getSelected() {
            return this.selectedPointIdx == -1? null : this.points[this.selectedPointIdx];
        }
        setSelected(point) {
            this.selectedPointIdx = this.points.findIndex(p => point == p);
            point.selected = true;
            this.dispatchEvent(new Event("select_point"));
        }
        moveSelected(x, y) {
            if(this.selectedPointIdx == -1)
                return;
            let p = this.points[this.selectedPointIdx];
            let idx = this.selectedPointIdx;
            let idx1 = this.selectedPointIdx - 1;
            let idx2 = this.selectedPointIdx + 1;

            let oldX = p.x;
            let oldY = p.y;
            let step = 0.01;
            if(idx != 0 && idx != this.points.length - 1)
            {
                if(x > this.points[idx1].x && x < this.points[idx2].x)
                    p.moveTo(x, y);
                else if(x <= this.points[idx1].x)
                    p.moveTo(this.points[idx1].x + step, y);
                else
                    p.moveTo(this.points[idx2].x - step, y);
            } 
            else if(idx == 0)
            {
                if(x < this.points[1].x)
                    p.moveTo(x, y);
                else
                    p.moveTo(this.points[1].x - step, y);
            } else {
                if(x > this.points[idx1].x)
                    p.moveTo(x, y);
                else
                    p.moveTo(this.points[idx1].x + step, y);
            }

            if(p.x != oldX || p.y != oldY)
                this.dispatchEvent(new Event("move_selected"));
        }
        addPoint(point) {
            this.points.push(point);
            this._sort();
            this.dispatchEvent(new Event("add_point"));
        }
        createPoint(x, y) {
            let point = new ctf.PointWithLimits(x, y);
            this.points.push(point);
            this._sort();
            this.dispatchEvent(new Event("create_point"));
            return point;
        }
        removeSelected() {
            if(this.selectedPointIdx != -1) {
                this.points.splice(this.selectedPointIdx, 1);
                this.selectedPointIdx = -1;
                this._sort();
                this.dispatchEvent(new Event("remove_point"));
            }
        }
        findByScreenCoordinates(x, y) {
            return this.points.find(p => this.helper.isPointerOnPoint(x, y, p), this);
        }
        find(predicate, thisArg) {
            return this.points.find(predicate, thisArg);
        }
        getSelectedPoint() {
            if(this.selectedPointIdx != -1)
                return this.points[this.selectedPointIdx];
        }
        contains(point) {
            return this.points.includes(point);
        } 
        repaint() {
            this.painter.clear();
            this.points.forEach(p => this.painter.drawPoint(p));
        }
        clear() {
            this.points = [];
            this.selectedPointIdx = -1;
        }

        // private
        _sort() {
            this.points = this.points.sort((p1, p2) => p1.x - p2.x);
        }
    }
    ctf.Scene = Scene;


    class OpacityScene extends Scene {
        constructor() {
            super();
            
            this.fixedPnt1 = new ctf.PointWithLimits(0, 1);
            this.fixedPnt1.setFixedX(0);
            this.fixedPnt1.setMinY(0);
            this.fixedPnt1.setMaxY(1);
            this.addPoint(this.fixedPnt1);

            this.fixedPnt2 = new ctf.PointWithLimits(1, 1);
            this.fixedPnt2.setFixedX(1);
            this.fixedPnt2.setMinY(0);
            this.fixedPnt2.setMaxY(1);
            this.addPoint(this.fixedPnt2);
        }
        addPoint(point) {
            this._sort();
            if(point.x == 0)
            {
                if(this.points.length != 0 && this.points[0].x == 0)
                    return;
                else {
                    point.setFixedX(0);
                    this.fixedPnt1 = point;
                }
            }
            else if(point.x == 1) 
            {
                if(this.points.length != 0 && this.points[this.points.length-1].x == 1)
                    return;
                else {
                    point.setFixedX(1);
                    this.fixedPnt2 = point;
                }
            }
            else 
            {
                point.setMinX(0);
                point.setMaxX(1);
            }
            point.setMinY(0);
            point.setMaxY(1);
            super.addPoint(point)
        }
        createPoint(x, y) {
            if(x <= 0 || x >= 1)
                return;
            if(y < 0 || y > 1)
                return;
            let point = new ctf.PointWithLimits(x, y);
            point.setMinX(0);
            point.setMaxX(1);
            point.setMinY(0);
            point.setMaxY(1);
            
            this.points.push(point);
            this._sort();
            this.dispatchEvent(new Event("create_point"));
            this.setSelected(point);
            return point;
        }
        removeSelected() {
            if(this.selectedPointIdx != 0 && this.selectedPointIdx != this.points.length - 1)
                super.removeSelected();
        }
        setDefault() {
            this._sort();
            if(this.selectedPointIdx != -1) {
                this.points[this.selectedPointIdx].selected = false;
                this.selectedPointIdx = -1;
            }
            this.fixedPnt1.y = 1;
            this.fixedPnt2.y = 1;
            this.points.splice(1, this.points.length - 2);
            this.dispatchEvent(new Event("set_default"));
        }
        repaint() {
            this.painter.clear();
            this.painter.drawCoordinateSystem();
            this._sort();
            this.points.forEach((p, i, arr) => {
                if(i != 0) {
                    let l = new ctf.GraphicsLine(arr[i-1], p);
                    this.painter.drawLine(l);
                }
                this.painter.drawPoint(p);
            });
        }

        // private
        _sort() {
            this.points = this.points.sort((p1, p2) => {
                if(p1 == this.fixedPnt1)
                    return -1;
                if(p2 == this.fixedPnt1)
                    return 1;
                if(p1 == this.fixedPnt2)
                    return 1;
                if(p2 == this.fixedPnt2)
                    return -1;
                return p1.x - p2.x;
            });
        }
    }
    ctf.OpacityScene = OpacityScene;


    class ColormapScene extends Scene {
        constructor() {
            super();
        }
        addPoint(point) {
            this._sort();
            point.setFixedY(0.5);
            if(point.x == 0) {
                if(this.points.length != 0 && this.points[0].x == 0)
                    return;
                else
                    point.setFixedX(0);
            }
            else if(point.x == 1) {
                if(this.points.length != 0 && this.points[this.points.length-1].x == 1)
                    return;
                else
                    point.setFixedX(1);
            }
            else {
                point.setMinX(0);
                point.setMaxX(1);
            }
            super.addPoint(point);
        }
        removeSelected() {
            if(this.selectedPointIdx == -1)
                return;
            if(this.points[this.selectedPointIdx].x == 0 || this.points[this.selectedPointIdx].x == 1)
                return;
            super.removeSelected();
        }
        createPoint(x, y) {
            if(x <= 0 || x >= 1)
                return;
            if(y < 0 || y > 1)
                return;
            let p = new ctf.PointWithLimits(x, y);
            this.points.push(p);
            this._sort();
            p.setMinX(0);
            p.setMaxX(1);
            p.setFixedY(0.5);

            this.dispatchEvent(new Event("create_point"));
            this.setSelected(p);
            return p;
        }
        repaint() {
            this.painter.clear();
            this.painter.drawCoordinateSystem();
            this._sort();
            this.points.forEach(p => this.painter.drawPoint(p));
        }
        setDefault() {
            this.points = [];
            let CTFunction = new ctf.Func();
            CTFunction.setDefault();
            CTFunction.colormapPoints.forEach((point, idx, arr) => {
                let p = new ctf.PointWithLimits(point.relativeVal, 0.5);
                p.setFixedY(0.5);
                if(idx == 0)
                    p.setFixedX(0);
                else if(idx == arr.length - 1)
                    p.setFixedX(1);
                this.addPoint(p);
            }, this);
            this.dispatchEvent(new Event("set_default"));
        }
        findByScreenCoordinates(x, y) {
            return super.findByScreenCoordinates(x, this.helper.toScreenY(0.5));
        }

        _sort() {
            this.points.sort((p1, p2) => p1.x - p2.x);
        }
    }
    ctf.ColormapScene = ColormapScene;
})()