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

    class MouseInterpreter {
        constructor() {
            this.mouseClicked = false;
            this.mousein = false;
            
            this.canvas = null;
            this.scene = null;
            this.helper = null;

            this.canvasPosition = { top: 0,
                                    left: 0 };
        }
        setCanvas(canvas) {
            let self = this;
            let jquery = canvas.jquery;
            jquery.mouseover(e => self.mousein = true);
            jquery.mouseleave(e => self.mousein = false);
            jquery.mousedown(e => {
                self.canvasPosition = {
                    top: e.pageY - e.offsetY,
                    left: e.pageX - e.offsetX
                };

                self.mouseClicked = true;
                self.onMouseClicked(e.offsetX, e.offsetY)
            });
            $(document).mouseup(e => {
                if(self.mouseClicked) {
                    let pos = self.canvasPosition;
                    let x = e.pageX - pos.left;
                    let y = e.pageY - pos.top;
                    self.mouseClicked = false;
                    self.onMouseReleased(x, y);
		}
            });
            $(document).mousemove(e => {
	       	if(self.mouseClicked) {
                    let pos = self.canvasPosition;
                    let x = e.pageX - pos.left;
                    let y = e.pageY - pos.top;
                    let dx = e.originalEvent.movementX;
                    let dy = e.originalEvent.movementY;
                    self.onMouseMoved(x, y, dx, dy);
		}
            });

        }
        setScene(scene) {
            this.scene = scene;
        }
        attachHelper(helper) {
            this.helper = helper;
        }
    }
    ctf.MouseInterpreter = MouseInterpreter;


    class OpacityMouseInterpreter extends MouseInterpreter {
        constructor() {
            super();
            this.draggedPoint = null;
            this.selectedPoint = null;
        }
        onMouseClicked(x, y) {
            if(this.selectedPoint)
                this.selectedPoint.selected = false;
            this.selectedPoint = null;
            let p = this.scene.findByScreenCoordinates(x, y);
            if(!p) {
                let sceneX = this.helper.coordinatesTransform.fromScreenX(x);
                let sceneY = this.helper.coordinatesTransform.fromScreenY(y);
                p = this.scene.createPoint(sceneX, sceneY);
            } else {
                this.scene.setSelected(p);
                this.scene.repaint();
            }
            this.selectedPoint = p;
            this.draggedPoint = p;
        }
        onMouseMoved(x, y, dx, dy) {
            if(this.draggedPoint) {
	        this.scene.moveSelected(
                    this.helper.fromScreenX(x), 
                    this.helper.fromScreenY(y));
                this.scene.repaint();
	    }
           
        }
        onMouseReleased(x, y) {
            if(this.draggedPoint) {
                this.draggedPoint = null;
                this.scene.dispatchEvent(new Event("change"));
            }
        }
    }
    ctf.OpacityMouseInterpreter = OpacityMouseInterpreter;


    class ColormapMouseInterpreter extends MouseInterpreter {
        constructor() {
            super();
            this.selectedPoint = null;
            this.draggedPoint = null;
        }
        onMouseClicked(x, y) {
            if(!this.mouseClicked)
                return;
            if(this.selectedPoint)
                this.selectedPoint.selected = false;
            this.selectedPoint = null;
            let p = this.scene.findByScreenCoordinates(x, y);
            if(!p) {
                let sceneX = this.helper.coordinatesTransform.fromScreenX(x);
                let sceneY = this.helper.coordinatesTransform.fromScreenY(y);

                p = this.scene.createPoint(sceneX, sceneY);
            } else {
                this.scene.setSelected(p);
                this.scene.repaint();
            }
            this.selectedPoint = p;
            this.draggedPoint = p;
        }
        onMouseMoved(x, y, dx, dy) {
            if(this.draggedPoint) {
                let x_ = this.helper.fromScreenX(x);
                let y_ = this.helper.fromScreenY(y);
                this.scene.moveSelected(x_, y_);
                this.scene.repaint();
	    }
        }
        onMouseReleased(x, y) { 
            if(this.draggedPoint) {
                this.draggedPoint = null;
                this.scene.dispatchEvent(new Event("change"));
            }
        }
    }
    ctf.ColormapMouseInterpreter = ColormapMouseInterpreter;
})()
