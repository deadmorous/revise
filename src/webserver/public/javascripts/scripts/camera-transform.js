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
    class TransformMatrix {
        constructor() {
            this.data = [];
            this.setDefault();
        }
        toArray() {
            return this.data;
        }
        setDefault() {
            this.data = new Array(16).fill(0);
            this.data[0] = this.data[5] = this.data[10] = this.data[15] = 1;
            this.data[14] = -30;
        }
        update() {
            let req = new objects.GetRequest("vs/camera/get", "");
            let dataArray = req.perform().receivedData();
            this.setDataArray(dataArray);
        }
        toArray() {
            return this.data;
        }
        setDataArray(array) {
            if(array.length != 16)
                throw new Error("The array length must equal 16");
            this.data = array;
        }
        getRotationMatrix() {
            let rotMatr = new objects.Matrix3();
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j)
                    rotMatr.set(i, j, this.get(i,j));
            return rotMatr;
        }
        setRotationMatrix(rotMatr) {
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j)
                    this.set(i, j, rotMatr.get(i,j));
        }
        rotate(angleRad, vector) {
            let currRotMatr = this.getRotationMatrix();
            let rotationMatr = new objects.RotationMatrix(angleRad, vector);
            currRotMatr = rotationMatr.dotMatr(currRotMatr);
            this.setRotationMatrix(currRotMatr);
        }
        getScreenNormal() {
            let k = new objects.Vector(0, 0, 1);
            let rotMatr = this.getRotationMatrix();
            return rotMatr.dotVec(k);
        }
        get(i, j) {
            return this.data[i * 4 + j];
        }
        set(i, j, value) {
            this.data[i * 4 + j] = value;
        }
    }
    let transformMatrix = new TransformMatrix();
    transformMatrix.update();
    
    let x_pos_JQ = $("#x-pos");
    let x_neg_JQ = $("#x-neg");
    let y_pos_JQ = $("#y-pos");
    let y_neg_JQ = $("#y-neg");
    let z_pos_JQ = $("#z-pos");
    let z_neg_JQ = $("#z-neg");
    let rotateClockwiseBtnJQ = $("#rot-clockwise");
    let rotateCounterBtnJQ = $("#rot-counterclockwise");

    function rotateDefault(angle, vec) {
        transformMatrix.setDefault();
        transformMatrix.rotate(angle, vec);

        let data = transformMatrix.toArray();
        let params = new URLSearchParams({
            m: JSON.stringify(data),
            cp: JSON.stringify([0, 0, 0])
        });
        let req = new objects.SetRequest("vs/camera/set", params.toString());
        req.perform(true);
    }
    function rotateClockwise() {
        transformMatrix.update();
        let screenNormal = transformMatrix.getScreenNormal();
        transformMatrix.rotate(Math.PI/6, screenNormal);
                
        let data = transformMatrix.toArray();
        let params = new URLSearchParams({
            m: JSON.stringify(data)
        });
        let req = new objects.SetRequest("vs/camera/set", params.toString());
        let async = true;
        req.perform(async);
    }
    function rotateCounterClockwise() {
        transformMatrix.update();
        let screenNormal = transformMatrix.getScreenNormal();
        transformMatrix.rotate(-Math.PI/6, screenNormal);
        
        let data = transformMatrix.toArray();
        let params = new URLSearchParams({
            m: JSON.stringify(data)
        });
        let req = new objects.SetRequest("vs/camera/set", params.toString());
        let async = true;
        req.perform(async);
    }

    z_pos_JQ.click( rotateDefault.bind(null, 0, new objects.Vector(0,0,0)) );
    z_neg_JQ.click( rotateDefault.bind(null, Math.PI, new objects.Vector(0,1,0)) );
    y_pos_JQ.click( rotateDefault.bind(null, -Math.PI/2, new objects.Vector(1,0,0)) );
    y_neg_JQ.click( rotateDefault.bind(null, Math.PI/2, new objects.Vector(1,0,0)) );
    x_pos_JQ.click( rotateDefault.bind(null, Math.PI/2, new objects.Vector(0,1,0)) );
    x_neg_JQ.click( rotateDefault.bind(null, -Math.PI/2, new objects.Vector(0,1,0)) );
    rotateClockwiseBtnJQ.click(rotateClockwise);
    rotateCounterBtnJQ.click(rotateCounterClockwise);
})()