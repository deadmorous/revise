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
    const black = new objects.Rgb(0, 0, 0);
    const white = new objects.Rgb(255, 255, 255);

    class ColorPicker extends EventTarget {
        constructor(jquery) {
            super();
            this.jquery = jquery;
            this.picker = new CP(jquery[0]);
            this.picker.self.classList.add("no-alpha");
            this.rgb = new objects.Rgb(0, 0, 0);
            this._init();
        }
        enable() {
            this.jquery.attr("disabled", false);
        }
        disable() {
            this.jquery.attr("disabled", true);
        }
        setColor(rgb) {
            this.rgb = rgb;

            let fontRgb = rgb.lengthTo(black) < rgb.lengthTo(white)?
                white : black;

            this.jquery.val(rgb.toSharpString());
            this.jquery.css("background-color", rgb.toString());
            this.jquery.css("color", fontRgb.toString());
            this.picker.set(rgb.r, rgb.g, rgb.b);
        }
        _init() {
            this.jquery.attr("readonly", true);
            this.picker.on('change', this._onChange.bind(this));
            this.picker.on('enter', this._onRaise.bind(this));
        }
        _onChange(r, g, b) {
            let rgb = new objects.Rgb(r, g, b);
            this.setColor(rgb);
            this.dispatchEvent(new Event("change"));
        }
        _onRaise() {
            let rgbStr = this.jquery.val();
            let rgb = objects.Rgb.fromSharpString(rgbStr);
            this.picker.set(rgb.r,
                            rgb.g,
                            rgb.b );
        }
    }
    objects.ColorPicker = ColorPicker;
})()