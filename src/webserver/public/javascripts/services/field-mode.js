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
    class FieldModeService {
        constructor() {
            this.selectJQ = $("#vis-mode-select");
            this.selectJQ.change( this.fieldModeChanged.bind(this) );
            
            this.activeUi = null;   // ui which must be shown in control panel
            this.defineActiveUi();
            this.stop();

            $(document).keydown(this, (e)=>{
                if(!this.activeUi || this.activeUi.isHidden)
                    return;
                if(e.key == "Enter") {
                    this.activeUi.submit();
                }
            });
        }
        run() {
            this.selectJQ.attr("disabled", false);
            this.activeUi.enable();
            this.restore();
        }
        stop() {
            this.selectJQ.attr("disabled", true);
            
            this.activeUi.hide();
            this.activeUi.disable();
        }
        restore() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            if(dataStr == null)
                return;
            let data = JSON.parse(dataStr);

            this.isRestoring = true;
            objects.isosurfaceUi.restore();
            objects.domainVoxelsUi.restore();
            objects.mipUi.restore();
            objects.argbLightUi.restore();
            objects.argbUi.restore();

            if(data.fieldMode) {
                this.selectJQ.val(data.fieldMode);
                this.selectJQ.change();
            }
            this.isRestoring = false;
        }
        store() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            let data;
            if(dataStr != null)
                data = JSON.parse(dataStr);
            else
                data = {};
            let fieldMode = this.selectJQ.val();
            data.fieldMode = fieldMode;
            localStorage.setItem(problem, JSON.stringify(data));
        }
        defineActiveUi() {
            switch (this.selectJQ.val()) {
                case "MIP":
                    this.activeUi = objects.mipUi;
                    break;
                case "Isosurface":
                    this.activeUi = objects.isosurfaceUi;
                    break;
                case "DomainVoxels":
                    this.activeUi = objects.domainVoxelsUi;
                    break;
                case "Argb":
                    this.activeUi = objects.argbUi;
                    break;
                case "ArgbLight":
                    this.activeUi = objects.argbLightUi;
                    break;
                default:
                    break;
            }
        }
        fieldModeChanged() {
            objects.clippingPlanesUi.cancel();
            objects.settingsService.hideDialog();

            this.activeUi.hide();   // firstly we hide old
            this.defineActiveUi();  // define new active ui according to the select
            this.activeUi.show();   // show new active ui
            this.store();
        }
        setPrimaryField(fieldInfo) {
            let range = new objects.Range(
                fieldInfo.range[0],
                fieldInfo.range[1],
                100 );

            this.setFieldRange(range);
            this.setEmptyUiValues();
        }

        setFieldRange(range) {
            objects.mipUi.setAcceptableInputValuesRange(range);
            objects.isosurfaceUi.setFieldRange(range);
            objects.domainVoxelsUi.setAcceptableInputValuesRange(range);
            objects.argbLightUi.setAcceptableInputValuesRange(range);
            objects.argbUi.setAcceptableInputValuesRange(range);
        }
        setFieldPresentationMode(mode) {
            objects.mipUi.setFieldPresentationMode(mode);
            objects.argbUi.setFieldPresentationMode(mode);
            objects.domainVoxelsUi.setFieldPresentationMode(mode);
            objects.argbLightUi.setFieldPresentationMode(mode);
            objects.isosurfaceUi.setFieldPresentationMode(mode);
        }
        setEmptyUiValues() {
            if(!objects.mipUi.isCompleted())
                objects.mipUi.reset();
            if(!objects.isosurfaceUi.isCompleted())
                objects.isosurfaceUi.reset();
            if(!objects.domainVoxelsUi.isCompleted())
                objects.domainVoxelsUi.reset();
            if(!objects.argbLightUi.isCompleted())
                objects.argbLightUi.reset();
            if(!objects.argbUi.isCompleted())
                objects.argbUi.reset();
        }
    }
    objects.fieldModeService = new FieldModeService();
})()
