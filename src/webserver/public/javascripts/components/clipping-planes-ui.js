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
    let coords = ["x", "y", "z"];
    let PlaneType = Object.freeze({
        points: 1,
        posAndNormal: 2
    });

    class Plane {
        constructor() {
            this.name = "";
        }
    }

    class PlaneUi extends Plane{
        constructor() {
            super();
            this.nodeJQ = $("<div>")
                .addClass("row-oriented");
            this.logoJQ = $("<div>")
                .addClass("logo")
                .appendTo(this.nodeJQ);
            this.captionJQ = $("<div>")
                .addClass("plane-name")
                .appendTo(this.nodeJQ);
        }
        setName(name) {
            this.name = name;
            this.captionJQ.text(name);
        }
        setFocused(focused = true) {
            if(focused)
                this.nodeJQ.addClass("focused");
            else 
                this.nodeJQ.removeClass("focused");
        }
        isFocused() {
            return this.nodeJQ.is(".focused");
        }
        remove() {
            this.nodeJQ.detach();
        }
    }

    class PlanePoints extends PlaneUi {
        constructor(point1, point2, point3) {
            super();

            let vec1 = point2.subtract(point1);
            let vec2 = point3.subtract(point1);
            if(vec1.crossProduct(vec2).length() == 0)
                throw new Error("Attempted to create the plane by three points lying on one line");

            this.point1 = point1;
            this.point2 = point2;
            this.point3 = point3;
            this.type = PlaneType.points;
            this.isInversed = false;

            this.nodeJQ.addClass("plane-points");
        }
        setInversed(isInversed = true) {
            this.isInversed = isInversed;
        }
        objectToSend() {
            let a = this.point2.subtract(this.point1);
            let b = this.point3.subtract(this.point1);
            let n = a.crossProduct(b).normalized();
            if( this.isInversed )
                n.inverse();
            return {
                "pos": this.point1.toArray(),
                "normal": n.toArray()
            }
        }
        getData() {
            let data = {};
            data.point1 = this.point1.getData();
            data.point2 = this.point2.getData();
            data.point3 = this.point3.getData();
            data.isInversed = this.isInversed;
            data.type = this.type;
            data.name = this.name;
            return data;
        }
        static makeByData(data) {
            let point1 = objects.Vector.makeByData(data.point1);
            let point2 = objects.Vector.makeByData(data.point2);
            let point3 = objects.Vector.makeByData(data.point3);
            let isInversed = data.isInversed;
            let plane = new PlanePoints(point1, point2, point3);
            plane.setInversed(isInversed);
            plane.setName(data.name);
            return plane;
        }
    }

    class PlanePosNormal extends PlaneUi {
        constructor(pos, normal) {
            if(normal.length() == 0)
                throw new Error("The normal must not have zero length");
            super();
            this.pos = pos;
            this.normal = normal;
            this.type = PlaneType.posAndNormal;

            this.nodeJQ.addClass("plane-pos-normal");
        }
        objectToSend() {
            return {
                "pos": this.pos.toArray(),
                "normal": this.normal.normalized().toArray()
            };
        }
        getData() {
            let data = {};
            data.pos = this.pos.getData();
            data.normal = this.normal.getData();
            data.type = this.type;
            data.name = this.name;
            return data;
        }
        static makeByData(data) {
            let pos = objects.Vector.makeByData(data.pos);
            let normal = objects.Vector.makeByData(data.normal);
            let plane = new PlanePosNormal(pos, normal);
            plane.setName(data.name);
            return plane;
        }
    }

    class FormPosAndNormal extends objects.Element {
        constructor() {
            super("clip-normal-origin");
            this.originInputs = {};
            this.normalInputs = {};
            for( let i = 0; i < 3; ++i ) {
                let c = coords[i];

                let selector1 = `div[name=originClip${c}]`;
                this.originInputs[c] = new objects.InputWithSlider($(selector1), c);

                let selector2 = `div[name=normalClip${c}]`;
                this.normalInputs[c] = new objects.InputWithSlider( $(selector2), c);
            }
        }
        getPlane() {
            let origin = new objects.Vector();
            let normal = new objects.Vector();
            for( let i = 0; i < 3; ++i ) {
                let c = coords[i];
                origin.set(i, this.originInputs[c].getValue());
                normal.set(i, this.normalInputs[c].getValue());
            }
            return new PlanePosNormal(origin, normal);
        }
        setPlane( plane ) {
            if( !(plane instanceof PlanePosNormal) )
                throw new Error("Passed argument must be an instance of PlanePosNormal");
            let normal = plane.normal;
            let origin = plane.pos;
            for( let i = 0; i < 3; ++i ) {
                let c = coords[i];
                this.normalInputs[c].setValue( normal.get(i) );
                this.originInputs[c].setValue( origin.get(i) );
            }
        }
        setAcceptableInputValuesRange(range) {
            for(let input in this.normalInputs)
                input.setRange(range);
            for(let input in this.originInputs)
                input.setRange(range);
        }
    }

    class FormThreePoints extends objects.Element {
        constructor() {
            super("clip-points");
            this.pointsInputs = [];
            for( let i = 0; i < 3; ++i ) {
                let pointInputs = {};
                for( let j = 0; j < 3; ++j ) {
                    let c = coords[j];
                    let selection = `div[name=clip-point${i+1}${c}]`;
                    pointInputs[c] = new objects.InputWithSlider($(selection), c);
                }
                this.pointsInputs.push(pointInputs);
            }
            this.isInversedJQ = $("#inverseNormal");
        }
        getPlane() {
            let points = [];
            for( let i = 0; i < 3; ++i ) {
                let point = new objects.Vector();
                let pointInputs = this.pointsInputs[i];
                for( let j = 0; j < 3; ++j ) {
                    let c = coords[j];
                    point.set(j, pointInputs[c].getValue());
                }
                points.push(point);
            }
            let plane = new PlanePoints(points[0], points[1], points[2]);
            plane.setInversed(this.isInversedJQ.is(":checked"));
            return plane;
        }
        setPlane(plane) {
            if( !(plane instanceof PlanePoints) )
                throw new Error("Failed to set plane: the plane must be the instance of PlanePoints");
            let points = [plane.point1, plane.point2, plane.point3];
            for( let i = 0; i < 3; ++i ) {
                let pointInputs = this.pointsInputs[i];
                let point = points[i];
                for( let j = 0; j < 3; ++j ) {
                    let c = coords[j];
                    pointInputs[c].setValue( point.get(j) );
                }
            }
            this.isInversedJQ.prop("checked", plane.isInversed);
        }
        setAcceptableInputValuesRange(range) {
            this.pointsInputs.forEach( (pointInputs)=>{
                for(let input in pointInputs)
                    input.setRange(range);
            });
        }
    }

    class ClipDialog extends objects.Dialog {
        constructor(){
            super("clip-dialog");
            this.formPosAndNormal = new FormPosAndNormal();
            this.formThreePoints = new FormThreePoints();
            this.activeForm = null;

            $("#clip-mode-select").change( this.update.bind(this));
        }
        show() {
            super.show();
            this.update();
        }
        update() {
            let clipMode = $("#clip-mode-select").val();
            if(clipMode == "origin-normal") {
                this.formPosAndNormal.show();
                this.activeForm = this.formPosAndNormal;
                this.formThreePoints.hide();
            }
            else {
                this.formThreePoints.show();
                this.activeForm = this.formThreePoints;
                this.formPosAndNormal.hide();
            }
        }
        getPlane() {
            let plane = this.activeForm.getPlane();
            let name = $("#clipping-plane-name").val();
            plane.setName(name);
            return plane;
        }
        setPlane(plane) {
            if(plane.type == PlaneType.points) {
                $("#clip-mode-select").val("points");
                this.formThreePoints.setPlane(plane);
            }
            else {
                $("#clip-mode-select").val("origin-normal");
                this.formPosAndNormal.setPlane(plane);
            }
            $("#clipping-plane-name").val(plane.name);
            this.update();
        }
        setAcceptableInputValuesRange(range) {
            this.formPosAndNormal.setAcceptableInputValuesRange(range);
            this.formThreePoints.setAcceptableInputValuesRange(range);
        }
        reset() {
            this.formPosAndNormal.reset();
            this.formThreePoints.reset();
        }
    }


    class PlanesContainer {
        constructor() {
            this.nodeJQ = $("#clipping-planes-list");
            this.planes = [];
        }
        add(plane) {
            this.planes.push(plane);
            this.nodeJQ.append(plane.nodeJQ);
        }
        remove(plane) {
            let index = this.planes.findIndex(p => plane == p);
            this.planes[index].remove();
            this.planes.splice(index, 1);
        }
        replace(oldPlane, newPlane) {
            let index = this.planes.findIndex(p => oldPlane == p);
            if(index == -1)
                return;
            oldPlane.nodeJQ.replaceWith(newPlane.nodeJQ);
            this.planes[index] = newPlane;
        }
        find(plane) {
            return this.planes.find(p => plane == p);
        }
        findByDom(planeDom) {
            return this.planes.find(p => p.nodeJQ[0] == planeDom);
        }
        map(callbackFn) {
            let a = this.planes.map(callbackFn);
            return this.planes.map(callbackFn);
        }
        setFocused(plane, isFocused = true) {
            if(isFocused) {
                let focused = this.getFocused();
                if(focused)
                    focused.setFocused(false);
            }
            plane.setFocused(isFocused);
        }
        getFocused() {
            return this.planes.find(plane => plane.isFocused());
        }
        clear() {
            this.planes.forEach(p => p.remove());
            this.planes = [];
        }
    }


    class ClippingPlanesUi extends objects.Element {
        constructor() {
            super("clipping-planes-UI");

            this.planesContainer = new PlanesContainer();
            this.dialog = new ClipDialog();

            this.okBtnJQ = this.nodeJQ.find("button[value=Ok]");
            this.addBtnJQ = $("#add-clip");
            this.removeBtnJQ = $("#remove-clip");
            this.editBtnJQ = $("#edit-clip");

            this.addBtnJQ.click( this.makeInDialog.bind(this) );
            this.removeBtnJQ.click( ()=>{
                let focused = this.planesContainer.getFocused();
                if(focused)
                    this.removePlane(focused);
            });
            this.editBtnJQ.click( ()=>{
                let focused = this.planesContainer.getFocused();
                if(focused)
                    this.editInDialog(focused);
            });

            this.okBtnJQ.click( this.submit.bind(this) );

            this.focusedPlane = null;
            this.disabled = false;
            
            this.hide();
        }
        makeInDialog() {
            this.addBtnJQ.blur();
            this.dialog.show();
            this.dialog.setTitle("Add clipping plane");

            let plane = null;
            let handler = ()=>{
                try {
                    plane = this.dialog.getPlane();
                    this.establishEventsFor(plane);
                    this.planesContainer.add(plane);
                    this.planesContainer.setFocused(plane);

                    this.onUpdate();
                } catch (error) {
                    if(plane && plane == this.focusedPlane)
                        this.removePlane();
                    throw error;
                }
            };
            this.dialog.addSubmitHandler( handler.bind(this) );
        }
        editInDialog(plane) {
            if(this.disabled)
                return;
            
            this.dialog.show();
            this.dialog.setTitle("Edit clipping plane");
            this.dialog.setPlane(plane);
            let onSubmit = ()=>{
                let newPlane = this.dialog.getPlane();
                this.establishEventsFor(newPlane);
                this.planesContainer.replace(plane, newPlane);
                this.planesContainer.setFocused(newPlane);
                this.onUpdate();
            };
            this.dialog.addSubmitHandler(onSubmit.bind(this));
        }
        removePlane(plane) {
            this.planesContainer.remove(plane);
            this.onUpdate();
        }
        onUpdate() {
            // send data to server
            let toSend = this.planesContainer.map(plane =>{
                return plane.objectToSend();
            });
            new objects.InputSetRequest("clippingPlanes", toSend).perform();

            // save to local storage
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            let data;
            if(dataStr == null)
                data = {};
            else
                data = JSON.parse(dataStr);
            let planesData = this.planesContainer.map(p => p.getData());
            data.clippingPlanes = planesData;
            localStorage.setItem(problem, JSON.stringify(data));
        }
        hideDialog() {
            if(!this.dialog.isHidden)
                this.dialog.cancel();
        }
        setPlanes(planesArray) {
            this.planesContainer.clear();
            planesArray.forEach(p => {
                this.establishEventsFor(p);
                this.planesContainer.add(p);
            }, this);
        }
        restore() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            if(dataStr == null)
                return;
            let data = JSON.parse(dataStr);
            let planesData = data.clippingPlanes;
            if(planesData == undefined || planesData.length == 0)
                return;
            let planes = planesData.map(pd => {
                let plane;
                if(pd.type == PlaneType.points)
                    plane = PlanePoints.makeByData(pd);
                else if(pd.type == PlaneType.posAndNormal)
                    plane = PlanePosNormal.makeByData(pd);
                return plane;
            });
            this.setPlanes(planes);
        }
        clearPlanes() {
            this.planesContainer.clear();
        }

        // overrided
        hide() {
            super.hide();
        }
        disable() {
            this.addBtnJQ.attr("disabled", true);
            this.editBtnJQ.attr("disabled", true);
            this.removeBtnJQ.attr("disabled", true);
            this.disabled = true;
        }
        enable() {
            this.addBtnJQ.attr("disabled", false);
            this.editBtnJQ.attr("disabled", false);
            this.removeBtnJQ.attr("disabled", false);
            this.disabled = false;
        }
        setAcceptableInputValuesRange(range) {
            this.dialog.setAcceptableInputValuesRange(range);
        }
        submit() {
            this.cancel();
        }
        cancel() {
            this.dialog.cancel();
            this.hide();
        }

        // private
        establishEventsFor(plane) {
            plane.nodeJQ.click(this, (e)=>{
                if(this.disabled)
                    return;
                let plane = this.planesContainer.findByDom(e.currentTarget);
                this.planesContainer.setFocused(plane);
            });
            plane.nodeJQ.dblclick( this.editInDialog.bind(this, plane) );
        }
    }
    objects.clippingPlanesUi = new ClippingPlanesUi();
})()