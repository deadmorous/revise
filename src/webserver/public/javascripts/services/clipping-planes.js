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
    class ClippingPlanesService {
        constructor() {
            this.btnJQ = $("#clip-btn");
            this.btnJQ.click( this.raiseUi.bind(this) );
            this.stop();

            objects.clippingPlanesUi.nodeJQ[0].addEventListener("hidden", ()=>{
                objects.controlPanel.setOwner( objects.fieldModeService.activeUi );
            });
        }
        run() {
            this.btnJQ.attr("disabled", false);
            objects.clippingPlanesUi.restore();
        }
        stop() {
            this.btnJQ.attr("disabled", true);
            objects.clippingPlanesUi.hide();
            objects.clippingPlanesUi.clearPlanes();
            objects.controlPanel.setOwner(objects.fieldModeService.activeUi);
        }
        // private
        raiseUi() {
            this.btnJQ.blur();
            objects.fieldModeService.activeUi.hide();
            objects.settingsService.hideDialog();
            
            objects.clippingPlanesUi.show();
        }
    }
    objects.clippingPlanesService = new ClippingPlanesService();
})()