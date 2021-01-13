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


    class Point {
        constructor(x, y) {
            this.x = x;
            this.y = y;
        }
        copy() {
            return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
        }
    }
    ctf.Point = Point;


    class GraphicsPoint extends Point {
        constructor(x, y) {
            super(x, y);
            this.selected = false;
        }
    }


    class MovablePoint extends GraphicsPoint {
        constructor(x, y) {
            super(x, y);
        }
        moveTo(x, y) {
            this.x = x;
            this.y = y;
        }
    }

    class PointWithLimits extends MovablePoint {
        constructor(x, y) {
            super(x, y);
            this.minX = null;
            this.maxX = null;
            this.minY = null;
            this.maxY = null;
        }
        setMinX(minX) {
            this.minX = minX;
            if(this.maxX != null && minX >= this.maxX)
                this.maxX = minX;
            if(this.x < minX)
                this.x = minX;
        }
        setMaxX(maxX) {
            this.maxX = maxX;
            if(this.minX != null && maxX <= this.minX)
                this.minX = maxX;
            if(this.x > maxX)
                this.x = maxX;
        }
        setMinY(minY) {
            this.minY = minY;
            if(this.maxY != null && minY >= this.maxY)
                this.maxY = minY;
            if(this.y < minY)
                this.y = minY;
        }
        setMaxY(maxY) {
            this.maxY = maxY;
            if(this.minY != null && maxY <= this.minY)
                this.minY = maxY;
            if(this.y > maxY)
                this.y = maxY;
        }
        setFixedX(x) {
            this.x = this.minX = this.maxX = x;
        }
        setFixedY(y) {
            this.y = this.minY = this.maxY = y;
        }
        moveTo(x, y) {
            let fitVal = (val, range) => {
                if(range[0] != null && range[1] != null) {
                    if(val >= range[0] && val <= range[1])
                        return val;
                    else if(val < range[0])
                        return range[0];
                    else if(val > range[1])
                        return range[1];
                }
                else if(range[0] != null) {
                    if(val > range[0])
                        return val;
                    else
                        return range[0];
                }
                else if(range[1] != null) {
                    if(val < range[1])
                        return val;
                    else
                        return range[1];
                }
                else
                    return val;
            };
            this.x = fitVal(x, [this.minX, this.maxX]);
            this.y = fitVal(y, [this.minY, this.maxY]);
        }
    }
    ctf.PointWithLimits = PointWithLimits;


    class Line {
        constructor(point1, point2) {
            this.point1 = point1;
            this.point2 = point2;
        }
        copy() {
            let p1 = this.point1.copy();
            let p2 = this.point2.copy();
            let res = Object.create(Object.getPrototypeOf(this));
            res.point1 = p1;
            res.point2 = p2;
            return res;
        }
    }


    class GraphicsLine extends Line {
        constructor(point1, point2) {
            super(point1, point2);
            this.color = null;
            this.opacity = 1;
            this.thickness = 1;
        }
        copy() {
            let c = super.copy();
            c.color = this.color;
            c.thickness = this.thickness;
            c.opacity = this.opacity;
            return c;
        }
    }
    ctf.GraphicsLine = GraphicsLine;


    class Rect {
        constructor(x, y, width, height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
        copy() {
            return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
        }
        leftBottom() {
            return new Point(this.x, this.y);
        }
        leftTop() {
            return new Point(this.x, this.y + this.height);
        }
        rightBottom() {
            return new Point(this.x + this.width, this.y);
        }
        rightTop() {
            return new Point(this.x + this.width, this.y + this.height);
        }
        left() {
            return this.x;
        }
        right() {
            return this.x + this.width;
        }
        top() {
            return this.y + this.height;
        }
        bottom() {
            return this.y;
        }
        center() {
            return new Point(
                this.x + this.width / 2,
                this.y + this.height / 2
            );
        }
        intersectsRect(otherRect) {
            return this.containsPoint(otherRect.leftBottom()) && 
                this.containsPoint(otherRect.rightBottom()) &&
                this.containsPoint(otherRect.leftTop()) &&
                this.containsPoint(otherRect.rightTop());
        }
        containsPoint(point) {
            if(point.x < this.x || point.x > this.x + this.width)
                return false;
            if(point.y < this.y || point.y > this.y + this.height)
                return false;
            return true;
        }
    }
    ctf.Rect = Rect;


    class GraphicsRect extends Rect {
        constructor(x, y, width, height) {
            super(x, y, width, height);
            this.color = new objects.Rgb();
        }
        copy() {
            let cp = super.copy();
            cp.color = new objects.Rgb(
                this.color.r,
                this.color.g,
                this.color.b
            );
            return cp;
        }
    }


    class FilledRect extends GraphicsRect {
        constructor(x, y, width, height) {
            super(x, y, width, height);
        }
    }


    class StrokeRect extends GraphicsRect {
        constructor(x, y, width, height) {
            super(x, y, width, height);
        }
    }


    class TextOptions {
        constructor() {
            this.size = 14;
            this.family = "Arial";
            this.color = "black";
            this.alignment = "center";
            this.baseline = "middle";
        }
        setSizeInPixels(size) {
            this.size = size;
        }
        setFontFamily(family) {
            this.family = family;
        }
        setColor(colorString) {
            this.color = colorString;
        }
        setAlignment(alignment) {
            this.alignment = alignment;
        }
        setBaseline(baseline) {
            this.baseline = baseline;
        }
        applyTo(canvasContext) {
            canvasContext.fillStyle = this.color;
            canvasContext.textAlign = this.alignment;
            canvasContext.textBaseline = this.baseline;
            canvasContext.font = `${this.size}px ${this.family}`;
        }
    }


    class PainterOptions {
        constructor() {
            this.linesColor = new objects.Rgb(0, 100, 0);
            this.pointOptions = {
                normalColor: new objects.Rgb(0, 0, 0),
                selectedColor: new objects.Rgb(0, 0, 0),
                outRectSize: 7,
                fillNormal: false,
                fillSelected: true
            };
            this.axles = {
                xAxleLength: 1.01,
                yAxleLength: 1.05,
                arrowEndSize: 10,
                labelsStepX: 0.2,
                labelsStepY: 0.25
            };
            this.textOptions = new TextOptions();
        }
        setDominantColor(rgb) {
            this.linesColor = rgb;
            this.pointOptions.normalColor = rgb;
            this.pointOptions.selectedColor = rgb;
            this.textOptions.color = rgb;
        }
    }


    class CtfPainterOptions extends PainterOptions {
        constructor() {
            super();
            this.nColormapRegions = 700;
        }
    }
    ctf.PainterOptions = CtfPainterOptions; 


    class CoordinatesTransform {
        constructor() {
            this.screenWidth = null;
            this.screenHeight = null;
            this.margin = {
                top: 30,
                bottom: 25,
                left: 40,
                right: 10
            };

            this.width = 1;
            this.height = 1;
            this.inverseY = true;
        }
        adaptToCanvas(canvas) {
            this.screenHeight = canvas.height;
            this.screenWidth = canvas.width;
        }
        toScreenX(x) {
            return this.margin.left + 
                (this.screenWidth - this.margin.left - this.margin.right) * x / this.width;
        }
        toScreenY(y) {
            let res = this.screenHeight - this.margin.top - this.margin.bottom;
            res *= y / this.height;
            if(this.inverseY)
                res = this.screenHeight - this.margin.bottom - res;
            else
                res += this.margin.top;
            return res;
        }
        toScreenSizeX(sizeX) {
            return sizeX * (this.screenWidth - this.margin.left - this.margin.right) / this.width;
        }
        toScreenSizeY(sizeY) {
            return sizeY * (this.screenHeight - this.margin.top - this.margin.bottom) / this.height;
        }
        fromScreenX(x) {
            let res = x - this.margin.left;
            return res / (this.screenWidth - this.margin.left - this.margin.right);
        }
        fromScreenY(y) {
            let res;
            if(this.inverseY)
                res = this.screenHeight - y - this.margin.bottom;
            else 
                res = y - this.margin.top;
            return res / (this.screenHeight - this.margin.bottom - this.margin.top);
        }
        fromScreenSizeX(sizeX) {
            return sizeX / (this.screenWidth - this.margin.left - this.margin.right) * this.width;
        }
        fromScreenSizeY(sizeY) {
            return sizeY / (this.screenHeight - this.margin.top - this.margin.bottom) * this.height;
        }
        fromScreenRect(rect) {
            let res = rect.copy();
            res.x = this.fromScreenX(rect.x);
            res.width = this.fromScreenSizeX(rect.width);
            res.height = this.fromScreenSizeY(rect.height);
            if(this.inverseY)
                res.y = this.fromScreenY(rect.y + rect.height);
            else
                res.y = this.fromScreenY(rect.y);
            return res;
        }
        toScreenPoint(point) {
            let copy = point.copy();
            copy.x = this.toScreenX(point.x);
            copy.y = this.toScreenY(point.y);
            return copy;
        }
        toScreenLine(line) {
            let cp = line.copy();
            cp.point1 = this.toScreenPoint(line.point1);
            cp.point2 = this.toScreenPoint(line.point2);
            return cp;
        }
        toScreenRect(rect) {
            let res = rect.copy();
            res.width = this.toScreenSizeX(rect.width);
            res.height = this.toScreenSizeY(rect.height);
            res.x = this.toScreenX(rect.x);
            if(this.inverseY)
                res.y = this.toScreenY(rect.y + rect.height);
            else
                res.y = this.toScreenX(rect.y);
            return res; 
        }
        setMarginTop(topMargin) {
            this.margin.top = topMargin;
        }
        setMarginBottom(bottomMargin) {
            this.margin.bottom = bottomMargin;
        }
        setMarginLeft(leftMargin) {
            this.margin.left = leftMargin;
        }
        setMarginRight(rightMargin) {
            this.margin.right = rightMargin;
        }
    }
    ctf.CoordinatesTransform = CoordinatesTransform;


    class Canvas {
        constructor(jquery) {
            this.jquery = jquery;
            this.context = jquery[0].getContext('2d');
            this.width = jquery.width();
            this.height = jquery.height();

            this.jquery[0].width = this.width;
            this.jquery[0].height = this.height;

            this.helper = null; 
        }
        setBackgroundColor(rgb) {
            let r = rgb.r;
            let g = rgb.g;
            let b = rgb.b;
            this.jquery.css("backgroundColor",`rgb(${r},${g},${b})`);
        }
        drawRect(rect) {
            if(rect instanceof FilledRect) {
                this.context.fillStyle = rect.color.toString();
                this.context.fillRect(rect.x, rect.y, rect.width, rect.height);
            } else if(rect instanceof StrokeRect) {
                this.context.strokeStyle = rect.color.toString();
                this.context.strokeRect(rect.x, rect.y, rect.width, rect.height);
            }
        }
        drawLine(line) {
            if(line instanceof GraphicsLine) {
                this.context.strokeStyle = line.color.toString();
                this.context.globalAlpha = line.opacity;
                this.context.beginPath();
                this.context.moveTo(line.point1.x, line.point1.y);
                this.context.lineTo(line.point2.x, line.point2.y);
                this.context.lineWidth = line.thickness;
                this.context.stroke();
            }
        }
        drawText(text, point, options) {
            if(options)
                options.applyTo(this.context);
            else
                options = new TextOptions();
            this.context.fillText(text, point.x, point.y);
        }
        clear() {
            let context = this.jquery[0].getContext("2d");
            context.clearRect(0, 0, this.width, this.height);
        }
    }
    ctf.Canvas = Canvas;


    class Painter {
        constructor(canvas) {
            this.canvas = canvas;
            this.painterOptions = null;
            this.coordinatesTransform = null;
        }
        setBackgroundColor(rgb) {
            this.canvas.setBackgroundColor(rgb);
            let black = new objects.Rgb(0, 0, 0);
            let white = new objects.Rgb(255, 255, 255);
            let lengthWhite = rgb.lengthTo(black);
            let lengthBlack = rgb.lengthTo(white);
            this.painterOptions.setDominantColor(lengthBlack > lengthWhite? white : black);
        }
        drawPoint(point) {
            let width = this.painterOptions.pointOptions.outRectSize;
            let height = width;
            let x = this.coordinatesTransform.toScreenX(point.x) - width / 2;
            let y = this.coordinatesTransform.toScreenY(point.y) - height / 2;
            let rect;
            if(!point.selected) {
                let toFill = this.painterOptions.pointOptions.fillSelected;
                if(!toFill)
                    rect = new FilledRect(x, y, width, height);
                else
                    rect = new StrokeRect(x, y, width, height);
                rect.color = this.painterOptions.pointOptions.normalColor;
            }
            else {
                let toFill = this.painterOptions.pointOptions.fillNormal;
                if(!toFill)
                    rect = new FilledRect(x, y, width, height);
                else
                    rect = new StrokeRect(x, y, width, height);
                rect.color = this.painterOptions.pointOptions.selectedColor;
            }
            this.canvas.drawRect(rect);
        }
        drawLine(line) {
            let screenLine = this.coordinatesTransform.toScreenLine(line);
            if(!screenLine.color)
                screenLine.color = this.painterOptions.linesColor;
            this.canvas.drawLine(screenLine);
        }
        drawRect(rect) {
            let screenRect = this.coordinatesTransform.toScreenRect(rect);
            this.canvas.drawRect(screenRect);
        }
        drawText(text, point, options) {
            if(!options)
                options = this.painterOptions.textOptions;
            this.canvas.drawText(
                text,
                this.coordinatesTransform.toScreenPoint(point),
                options
            );
        }
        clear() {
            if(this.canvas)
                this.canvas.clear();
        }
    }


    class CtfPainter extends Painter {
        constructor(canvas) {
            super(canvas);
            this.interpolate = new objects.Ctf.Interpolate();
        }
        attachColormapPoints(points) {
            this.interpolate.attachColormapPoints(points)
        }
    }
    ctf.BasePainter = CtfPainter;


    class OpacityPainter extends CtfPainter {
        constructor(canvas) {
            super(canvas);
        }
        attachColormapPoints(colormapPoints) {
            this.interpolate.attachColormapPoints(colormapPoints);
        }
        attachOpacityPoints(opacityPoints) {
            this.interpolate.attachOpacityPoints(opacityPoints);
        }
        drawCoordinateSystem() {
            this._drawColormap();
            this._drawAxles();
            this._drawLabels();
        }

        // private
        _drawAxles() {
            let origin = new Point(0,0);
            let xAxleEnd = new Point(this.painterOptions.axles.xAxleLength, 0);
            let yAxleEnd = new Point(0, this.painterOptions.axles.yAxleLength);
            this.drawLine(new GraphicsLine(origin, xAxleEnd));
            this.drawLine(new GraphicsLine(origin, yAxleEnd));
        }
        _drawLabels() {
            let originTextOptions = this.painterOptions.textOptions;
            originTextOptions.setAlignment("right");
            originTextOptions.setBaseline("top");
            this.drawText("0", new Point(0,0), originTextOptions);

            // y-labels
            {
                let yAxleLabelsOptions = this.painterOptions.textOptions;
                yAxleLabelsOptions.setBaseline("middle");
                yAxleLabelsOptions.setAlignment("right");
                
                let stepY = this.painterOptions.axles.labelsStepY;
                let x1 = -this.coordinatesTransform.fromScreenSizeX(5);
                let x2 = -x1;
                let x1_ = 2 * x1;
                for(let i = stepY; i < this.painterOptions.axles.yAxleLength; i += stepY) 
                {
                    let y = i;
                    let p1 = new Point(x1, y);
                    let p2 = new Point(x2, y);
                    let l = new GraphicsLine(p1, p2);
                    this.drawLine(l);

                    this.drawText(
                        Number(i).toPrecision(2),
                        new Point(x1_, i),
                        this.painterOptions.textOptions);
                }
            }

            // x-labels
            {
                let xAxleLabelsOptions = this.painterOptions.textOptions;
                xAxleLabelsOptions.setBaseline("top");
                xAxleLabelsOptions.setAlignment("center");
                
                let stepX = this.painterOptions.axles.labelsStepX;
                let y1 = this.coordinatesTransform.fromScreenSizeY(5);
                let y2 = -y1;
                let y1_ = - 2 * y1; 
                for(let i = stepX; i < this.painterOptions.axles.xAxleLength; i += stepX) 
                {
                    let x = i;
                    let p1 = new Point(x, y1);
                    let p2 = new Point(x, y2);
                    let l = new GraphicsLine(p1, p2);
                    l.color = this.painterOptions.linesColor;
                    this.drawLine(l);

                    this.drawText(
                        Number(i).toPrecision(1),
                        new Point(i, y1_),
                        xAxleLabelsOptions);
                }
            }
        }
        _drawColormap() {
            if(!this.interpolate.colormapPoints)
                return;
            let nRegions = this.painterOptions.nColormapRegions;
            let step = this.coordinatesTransform.width / nRegions;
            let x;
            for(x = 0 + step / 2; x <= 1; x += step) {
                let y = this.interpolate.interpolateOpacity(x);
                let l = new GraphicsLine(
                    new Point(x, 0),
                    new Point(x, y)
                );
                l.thickness = this.coordinatesTransform.toScreenSizeX(step);
                l.opacity = y;
                l.color = this.interpolate.interpolateColor(
                    x / this.coordinatesTransform.width);
                this.drawLine(l);
            }
        }
    }
    ctf.OpacityPainter = OpacityPainter;


    class ColormapPainter extends CtfPainter {
        constructor(canvas) {
            super(canvas);
        }
        drawCoordinateSystem() {
            this._drawColormap();
            this._drawAxles();
            this._drawLabels();
        }
        attachColormapPoints(colormapPoints) {
            this.interpolate.attachColormapPoints(colormapPoints);
        }

        // private
        _drawColormap() {
            if(!this.interpolate.colormapPoints)
                return;

            let nRegions = this.painterOptions.nColormapRegions;
            let step = this.coordinatesTransform.width / nRegions;
            let x;
            for(x = 0 + step / 2; x <= 1; x += step) {
                let l = new GraphicsLine(
                    new Point(x, 0),
                    new Point(x, 1)
                );
                l.thickness = this.coordinatesTransform.toScreenSizeX(step);
                l.color = this.interpolate.interpolateColor(
                    x / this.coordinatesTransform.width);
                this.drawLine(l);
            }
        }
        _drawAxles() {
            let origin = new Point(0, 0);
            let axleEnd = new Point(1, 0);
            let l = new GraphicsLine(origin, axleEnd);
            l.thickness = 1;
            this.drawLine(l);
        }
        _drawLabels() {
            let stepX = this.painterOptions.axles.labelsStepX;
            let x = 0;
            let labelTextOpts = this.painterOptions.textOptions;
            labelTextOpts.alignment = "center";
            labelTextOpts.baseline = "top";

            let arrowSize = this.painterOptions.axles.arrowEndSize;
            let y1 = this.coordinatesTransform.fromScreenSizeY(arrowSize) / 2;
            let y2 = -this.coordinatesTransform.fromScreenSizeY(arrowSize) / 2;
            for(; x < this.painterOptions.axles.xAxleLength; x += stepX) {
                let x_ = this.coordinatesTransform.toScreenX(x);
                let p = new Point(x, y2 * 1.8);
                this.drawText(
                    Number(x).toPrecision(1),
                    p,
                    labelTextOpts
                );

                let labelLine = new GraphicsLine(
                    new Point(x, y1),
                    new Point(x, y2)
                );
                this.drawLine(labelLine);
            }
        }
    }
    ctf.ColormapPainter = ColormapPainter;
})()