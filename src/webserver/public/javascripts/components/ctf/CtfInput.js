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

    class CtfElementInput extends EventTarget {
        constructor(canvasJquery) {
            super();
            this.canvas = new ctf.Canvas(canvasJquery);

            let coordinatesTransform = this._makeCoordinateTransform(this.canvas);
            let painterOptions = this._makePainterOptions();
            this.helper = this._makeHelper(coordinatesTransform, painterOptions);
            this.painter = this._makePainter(this.canvas, coordinatesTransform, painterOptions);
            this.scene = this._makeScene(this.canvas, this.painter, this.helper);
            this.mouseInterpreter = this._makeMouseInterpreter(this.canvas, this.scene, this.helper);
        }
        setBackgroundColor(rgb) {
            this.painter.setBackgroundColor(rgb);
            this.scene.repaint();
        }

        // private
        _makePainterOptions() {
            return new ctf.PainterOptions();
        }
        _makeCoordinateTransform(canvas) {
            let coordinatesTransform = new ctf.CoordinatesTransform();
            coordinatesTransform.adaptToCanvas(canvas);
            return coordinatesTransform;
        }
        _makeHelper(coordinatesTransform, painterOptions) {
            let helper = new ctf.CanvasHelper();
            helper.coordinatesTransform = coordinatesTransform;
            helper.painterOptions = painterOptions;
            return helper;
        }
        _makePainter(canvas, coordinatesTransform, painterOptions) {
            let painter = new ctf.BasePainter(canvas);
            painter.coordinatesTransform = coordinatesTransform;
            painter.painterOptions = painterOptions;
            return painter;
        }
        _makeScene(canvas, painter, helper) {
            let scene = new ctf.Scene();
            scene.canvas = canvas;
            scene.painter = painter;
            scene.helper = helper;
            return scene;
        }
        _makeMouseInterpreter(canvas, scene, helper) {
            let mouseInterpreter = new ctf.MouseInterpreter();
            mouseInterpreter.setCanvas(canvas);
            mouseInterpreter.setScene(scene);
            mouseInterpreter.helper = helper;
            return mouseInterpreter;
        }
    }


    class OpacityInput extends CtfElementInput {
        constructor() {
            super($("#ctf-opacity-canvas"));

            this.relvalNumbInp = new objects.NumberInput($("#ctf-opacity-selected-relval"), "");
            this.opacityNumbInp = new objects.NumberInput($("#ctf-opacity-selected-opacity"), "");
            this._init();
        }
        attachColormapPoints(colormapPoints) {
            this.painter.attachColormapPoints(colormapPoints);
            this.scene.repaint();
        }
        getPoints() {
            return this.scene.points.map(sp => new ctf.OpacityPoint(sp.x, sp.y));
        }
        setPoints(points) {
            this.scene.clear();
            points.forEach(p => {
                this.scene.addPoint(new ctf.PointWithLimits(p.relativeVal, p.opacity));
            });
            
            this.relvalNumbInp.reset();
            this.opacityNumbInp.reset();

            this.relvalNumbInp.disable();
            this.opacityNumbInp.disable();        
        }
        setDefault() {
            this.relvalNumbInp.reset();
            this.opacityNumbInp.reset();

            this.relvalNumbInp.disable();
            this.opacityNumbInp.disable();

            this.scene.setDefault();
            
            this.painter.attachOpacityPoints(this.getPoints());
            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }

        //private
        _init() {
            this.relvalNumbInp.disable();
            this.opacityNumbInp.disable();

            this.scene.repaint();
            
            let self = this;
            $("#ctf-opacity-remove-btn").click(e => self.scene.removeSelected());
            $("#ctf-opacity-setDefault-btn").click(e => self.setDefault());

            this.scene.addEventListener("select_point", this._onSelectPoint.bind(this));
            this.scene.addEventListener("add_point", this._onAddPoint.bind(this));
            this.scene.addEventListener("move_selected", this._onMovePoint.bind(this));
            this.scene.addEventListener("remove_point", this._onRemovePoint.bind(this));
            this.scene.addEventListener("create_point", this._onCreatePoint.bind(this));

            this.relvalNumbInp.addEventListener("change", this._onAnyInputChanged.bind(this));
            this.opacityNumbInp.addEventListener("change", this._onAnyInputChanged.bind(this));
        }
        _makeCoordinateTransform(canvas) {
            let coodrdsTransform = super._makeCoordinateTransform(canvas);
            coodrdsTransform.setMarginTop(10);
            return coodrdsTransform;
        }
        _makeHelper(coordinatesTransform, painterOptions) {
            return super._makeHelper(coordinatesTransform, painterOptions);
        }
        _makePainterOptions() {
            let po = new ctf.PainterOptions();
            po.axles.yAxleLength = 1.01;
            return po;
        }
        _makePainter(canvas, coordinatesTransform, painterOptions) {
            let painter = new ctf.OpacityPainter(canvas);
            painter.coordinatesTransform = coordinatesTransform;
            painter.painterOptions = painterOptions;
            return painter;
        }
        _makeScene(canvas, painter, helper) {
            let scene = new ctf.OpacityScene();
            scene.canvas = canvas;
            scene.painter = painter;
            scene.helper = helper;
            return scene;
        }
        _makeMouseInterpreter(canvas, scene, helper) {
            let mouseInterpreter = new ctf.OpacityMouseInterpreter();
            mouseInterpreter.setCanvas(canvas);
            mouseInterpreter.setScene(scene);
            mouseInterpreter.helper = helper;
            return mouseInterpreter;
        }

        // slots
        _onSelectPoint() {
            let selectedPt = this.scene.getSelected();

            this.relvalNumbInp.setValue(selectedPt.x);
            this.opacityNumbInp.setValue(selectedPt.y);

            this.relvalNumbInp.enable();
            this.opacityNumbInp.enable();

            this.scene.repaint();
        }
        _onRemovePoint() {
            this.relvalNumbInp.reset();
            this.opacityNumbInp.reset();

            this.relvalNumbInp.disable();
            this.opacityNumbInp.disable();
            this.painter.attachOpacityPoints(this.getPoints());
            
            this.scene.repaint();

            this.dispatchEvent(new Event("change"));
        }
        _onAddPoint() {
            this.painter.attachOpacityPoints(this.getPoints());

            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        _onMovePoint() {
            let selectedPt = this.scene.getSelected();
            this.opacityNumbInp.setValue(selectedPt.y);
            this.relvalNumbInp.setValue(selectedPt.x);
            this.painter.attachOpacityPoints(this.getPoints());
            this.scene.repaint();
        }
        _onCreatePoint() {
            this.painter.attachOpacityPoints(this.getPoints());
            this.scene.repaint();
        }
        _onAnyInputChanged() {
            let selectedPt = this.scene.getSelected();
            if(!this.relvalNumbInp.isCompleted()) {
                this.relvalNumbInp.setValue(selectedPt.x);
                return;
            }
            if(!this.opacityNumbInp.isCompleted()) {
                this.opacityNumbInp.setValue(selectedPt.y);
                return;
            }
            let x = this.relvalNumbInp.getValue();
            let y = this.opacityNumbInp.getValue();
            this.scene.moveSelected(x, y);
            
            this.relvalNumbInp.setValue(selectedPt.x);
            this.opacityNumbInp.setValue(selectedPt.y);
        }
    }
    ctf.OpacityInput = OpacityInput;


    class Pair {
        constructor(scenePoint, colormapPoint) {
            this.scenePoint = scenePoint;
            this.colormapPoint = colormapPoint;
        }
        xChanged() {
            this.colormapPoint.relativeVal = this.scenePoint.x;
        }
        static fromColormapPoint(colormapPoint) {
            let scPt = new ctf.PointWithLimits(colormapPoint.relativeVal, 0.5);
            return new Pair(scPt, colormapPoint);
        }
    }

    class ColormapInput extends CtfElementInput {
        constructor() {
            super($("#ctf-colormap-canvas"));

            this.pairs = [];
            this.selectedPair = null;

            this.relValNumbInp = new objects.NumberInput("#ctf-colormap-relval-input");
            this.colorPicker = new objects.ColorPicker($("#ctf-colormap-color-input"));

            this._init();
        }
        setPoints(colormapPoints) {
            this.scene.clear();
            this.pairs = colormapPoints.map(cmPt => {
                let pair = Pair.fromColormapPoint(cmPt);
                this.scene.addPoint(pair.scenePoint);
                return pair;
            }, this);
            
            this.painter.attachColormapPoints(colormapPoints);
            this.selectedPair = null;

            this.colorPicker.disable();

            this.relValNumbInp.disable();
            this.relValNumbInp.reset();

            this.selectedPair = null;

            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        getPoints() {
            return this.pairs.map(connection => connection.colormapPoint);
        }
        setDefault() {
            this.colorPicker.disable();
            this.relValNumbInp.disable();
            this.relValNumbInp.reset();

            let ctf = new objects.Ctf.Func();
            ctf.setDefault();
            this.setPoints(ctf.colormapPoints);

            this.selectedPair = null;
        }
        
        // private
        _init() {
            this.scene.repaint();
            this.relValNumbInp.disable();

            $("#ctf-colormap-setDefault-btn").click(e => this.setDefault());
            $("#ctf-colormap-remove-btn").click(e => this.scene.removeSelected());

            this.scene.addEventListener("select_point", this._onSelectPoint.bind(this));
            this.scene.addEventListener("add_point", this._onAddPoint.bind(this));
            this.scene.addEventListener("move_selected", this._onMovePoint.bind(this));
            this.scene.addEventListener("remove_point", this._onRemovePoint.bind(this));
            this.scene.addEventListener("create_point", this._onCreatePoint.bind(this));
            this.scene.addEventListener("select_point", this._onSelectPoint.bind(this));

            this.colorPicker.addEventListener("change", this._onColorPickerChanged.bind(this));
            this.relValNumbInp.addEventListener("change", this._onRelativeValInputChanged.bind(this));
        }

        _makePainterOptions() {
            return new ctf.PainterOptions();
        }
        _makeCoordinateTransform(canvas) {
            let coordinatesTransform = new ctf.CoordinatesTransform();
            coordinatesTransform.adaptToCanvas(canvas);
            coordinatesTransform.setMarginBottom(25);
            coordinatesTransform.setMarginTop(10);
            return coordinatesTransform;
        }
        _makePainter(canvas, coordinatesTransform, painterOptions) {
            let painter = new ctf.ColormapPainter(canvas);
            painter.coordinatesTransform = coordinatesTransform;
            painter.painterOptions = painterOptions;
            return painter;
        }
        _makeMouseInterpreter(canvas, scene, helper) {
            let mouseInterpreter = new ctf.ColormapMouseInterpreter();
            mouseInterpreter.setCanvas(canvas);
            mouseInterpreter.setScene(scene);
            mouseInterpreter.helper = helper;
            return mouseInterpreter;
        }
        _makeScene(canvas, painter, helper) {
            let scene = new ctf.ColormapScene();
            scene.canvas = canvas;
            scene.painter = painter;
            scene.helper = helper;
            return scene;
        }

        // slots
        _onSelectPoint() {
            this.colorPicker.enable();

            this.relValNumbInp.enable();

            let scPt = this.scene.getSelected();

            this.selectedPair = this.pairs.find(pair => pair.scenePoint == scPt);

            let relVal = scPt.x;
            let rgb = this.selectedPair.colormapPoint.rgb;
            let rgbStr = rgb.toString();

            this.relValNumbInp.setValue(relVal);
            this.colorPicker.setColor(rgb);

            this.scene.repaint();
        }
        _onRemovePoint() {
            let idx = this.pairs.indexOf(this.selectedPair);
            this.pairs.splice(idx, 1);
            this.painter.attachColormapPoints(this.getPoints());
            this.scene.repaint();

            this.colorPicker.disable();
            this.relValNumbInp.disable();
            this.relValNumbInp.reset();
            this.dispatchEvent(new Event("change"));
        }
        _onAddPoint() {
            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        _onMovePoint() {
            this.selectedPair.xChanged();
            this.relValNumbInp.setValue(this.selectedPair.scenePoint.x);
            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        _onCreatePoint() {
            let currColormapPts = this.getPoints();
            let interpolate = new ctf.Interpolate(null, currColormapPts);

            this.pairs = this.pairs.sort((p1, p2) => p1.scenePoint.x - p2.scenePoint.x);
            let createdScPt = this.scene.points.find((scPt, idx) => {
                return scPt.x != this.pairs[idx].scenePoint.x;
            }, this);
            let rgb = interpolate.interpolateColor(createdScPt.x);
            let cmPt = new ctf.ColormapPoint(createdScPt.x, rgb);
            let pair = new Pair(createdScPt, cmPt);
            this.pairs.push(pair);

            this.painter.attachColormapPoints(this.getPoints());
            
            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        _onColorPickerChanged() {
            let rgb = this.colorPicker.rgb;

            if(!this.selectedPair)
                return;
            this.selectedPair.colormapPoint.rgb = rgb;

            this.scene.repaint();
            this.dispatchEvent(new Event("change"));
        }
        _onRelativeValInputChanged() {
            if(!this.relValNumbInp.isCompleted()) {
                this.relValNumbInp.setValue(this.selectedPair.scenePoint.x);
                return;
            }
            let newVal = this.relValNumbInp.getValue();
            this.scene.moveSelected(newVal, 0.5);
            newVal = this.selectedPair.scenePoint.x;
            this.relValNumbInp.setValue(newVal);
        }
    }
    ctf.ColormapInput = ColormapInput;
})()