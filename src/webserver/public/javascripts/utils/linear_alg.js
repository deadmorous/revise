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
    class Vector {
        constructor(x, y, z) {
            this.data = [x, y, z];
        }
        get(pos) {
            return this.data[pos];
        }
        set(pos, value) {
            this.data[pos] = value;
        }
        subtract(other) {
            return new Vector(
                this.get(0) - other.get(0),
                this.get(1) - other.get(1),
                this.get(2) - other.get(2) );
        }
        crossProduct(other) {
                return new Vector(
                    this.get(1) * other.get(2) - this.get(2) * other.get(1),
                    this.get(2) * other.get(0) - this.get(0) * other.get(2),
                    this.get(0) * other.get(1) - this.get(1) * other.get(0)
                );
        }
        normalize() {
            let length = this.length();
            this.data[0] /= length;
            this.data[1] /= length;
            this.data[2] /= length;
        }
        normalized() {
            let result = new Vector(this.data[0], this.data[1], this.data[2]);
            result.normalize();
            return result;
        }
        skewMatrix() {
            let x = this.get(0);
            let y = this.get(1);
            let z = this.get(2);
            let array = 
                [ 0, -z,  y,
                  z,  0, -x,
                 -y,  x,  0];
            let result = new Matrix3();
            result.data = array;
            return result;
        }
        tensorProd(other) {
            let result = new Matrix3();
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j)
                    result.set(i, j, this.get(i) * other.get(j));
            return result;
        }
        toArray() {
            return this.data;
        }
        inverse() {
            this.data[0] = - this.data[0];
            this.data[1] = - this.data[1];
            this.data[2] = - this.data[2];
        }
        length() {
            return Math.sqrt( 
                this.data[0] * this.data[0] + 
                this.data[1] * this.data[1] +
                this.data[2] * this.data[2] );
        }
        getData() {
            return {
                x: this.data[0],
                y: this.data[1],
                z: this.data[2]
            };
        }
        static makeByData(data) {
            return new Vector(data.x, data.y, data.z);
        }
    }
    objects.Vector = Vector;


    class Matrix3 {
        constructor() {
            this.data = new Array(9).fill(0);
        }
        get(row, col) {
            return this.data[row * 3 + col];
        }
        set(row, col, value) {
            this.data[row * 3 + col] = value;
        }
        plus(other) {
            let result = new Matrix3();
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j)
                    result.set(i, j, this.get(i,j) + other.get(i,j));
            return result;
        }
        minus(other) {
            let result = new Matrix3();
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j)
                    result.set(i, j, this.get(i,j) - other.get(i,j));
            return result;
        }
        dotMatr(other) {
            let result = new Matrix3();
            for(let i = 0; i < 3; ++i)
                for(let j = 0; j < 3; ++j) {
                    let buf = 0;
                    for(let k = 0; k < 3; ++k)
                        buf += this.get(i, k) * other.get(k, j);
                    result.set(i, j, buf);
                }
            return result;
        }
        dotVec(vec) {
            let result = new Vector();
            for(let i = 0; i < 3; ++i) {
                let buf = 0; 
                for(let j = 0; j < 3; ++j) 
                    buf += vec.get(j) * this.get(i, j);
                result.set(i, buf);
            }
            return result;
        }
        multiplyNumber(factor) {
            let result = new Matrix3();
            result.data = this.data.map( value => value * factor );
            return result;
        }
        static eye() {
            let result = new Matrix3();
            for(let i = 0; i < 3; ++i)
                result.set(i, i, 1);
            return result;
        }
    }
    objects.Matrix3 = Matrix3;


    class RotationMatrix extends Matrix3 {
        constructor(angleRad, vector) {
            super();
            let add1 = vector.tensorProd(vector);
            let add2 = Matrix3.eye()
                .minus(add1)
                .multiplyNumber(Math.cos(angleRad));
            let add3 = vector.skewMatrix()
                .multiplyNumber(Math.sin(angleRad));
            let buf = add1
                        .plus(add2)
                        .plus(add3);
            this.data = buf.data;
        }
    }
    objects.RotationMatrix = RotationMatrix;
})()