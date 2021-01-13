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
    class CameraTransformService {
        constructor() {
            this.x_pos_JQ = $("#x-pos");
            this.x_neg_JQ = $("#x-neg");
            this.y_pos_JQ = $("#y-pos");
            this.y_neg_JQ = $("#y-neg");
            this.z_pos_JQ = $("#z-pos");
            this.z_neg_JQ = $("#z-neg");
            this.rotateClockwiseBtnJQ = $("#rot-clockwise");
            this.rotateCounterBtnJQ = $("#rot-counterclockwise");

            this.stop();
        }
        run() {
            this.x_pos_JQ.attr("disabled", false);
            this.x_neg_JQ.attr("disabled", false);
            this.y_pos_JQ.attr("disabled", false);
            this.y_neg_JQ.attr("disabled", false);
            this.z_pos_JQ.attr("disabled", false);
            this.z_neg_JQ.attr("disabled", false);
            this.rotateClockwiseBtnJQ.attr("disabled", false);
            this.rotateCounterBtnJQ.attr("disabled", false);
            this.z_pos_JQ.click();
        }
        stop() {
            this.x_pos_JQ.attr("disabled", true);
            this.x_neg_JQ.attr("disabled", true);
            this.y_pos_JQ.attr("disabled", true);
            this.y_neg_JQ.attr("disabled", true);
            this.z_pos_JQ.attr("disabled", true);
            this.z_neg_JQ.attr("disabled", true);
            this.rotateClockwiseBtnJQ.attr("disabled", true);
            this.rotateCounterBtnJQ.attr("disabled", true);
        }
    }
    objects.cameraTransformService = new CameraTransformService();
})()