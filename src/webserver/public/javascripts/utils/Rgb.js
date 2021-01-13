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
    class Rgb {
        constructor(r, g, b) {
            this.r = r? r : 0;
            this.g = g? g : 0;
            this.b = b? b : 0;
        }
        toString() {
            return `rgb(${this.r},${this.g},${this.b})`;
        }
        toSharpString() {
            let r = Math.round(this.r).toString(16);
            if(r.length == 1)   
                r = `0${r}`;

            let g = Math.round(this.g).toString(16);
            if(g.length == 1)   
                g = `0${g}`;

            let b = Math.round(this.b).toString(16);
            if(b.length == 1)   
                b = `0${b}`;
            return `#${r}${g}${b}`;
        }
        lengthTo(other) {
            return Math.sqrt(
                (other.r - this.r) * (other.r - this.r) + 
                (other.g - this.g) * (other.g - this.g) + 
                (other.b - this.b) * (other.b - this.b)
            );
        }
        static fromSharpString(sharpString) {
            let r = Number("0x" + sharpString.slice(1,3));
            let g = Number("0x" + sharpString.slice(3,5));
            let b = Number("0x" + sharpString.slice(5,7));
            return new Rgb(r, g, b);
        }
        static makeFromTuple(rgb) {
            return new Rgb( rgb.r,
                            rgb.g,
                            rgb.b );
        }
    }
    objects.Rgb = Rgb;
})()