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

    class SettingsData {
        constructor() {
            this.fovY = 30;
            this.renderPatience = 25;
            this.renderQuality = 10;
            this.fieldPresentationMode = "absolute";
            this.animationInterval = 500;
            this.backgroundColor = new objects.Rgb(0, 0, 0);
        }

        getData() {
            let data = {};
            data.fovY = this.fovY;
            data.renderQuality = this.renderQuality;
            data.renderPatience = this.renderPatience;
            data.fieldPresentationMode = this.fieldPresentationMode;
            data.animationInterval = this.animationInterval;
            data.backgroundColor = this.backgroundColor;
            return data;
        }
        static make(data) {
            let settingsData = new SettingsData();
            settingsData.fieldPresentationMode = data.fieldPresentationMode;
            settingsData.fovY = data.fovY;
            settingsData.renderPatience = data.renderPatience;
            settingsData.renderQuality = data.renderQuality;
            settingsData.animationInterval = data.animationInterval;
            settingsData.backgroundColor = objects.Rgb
                .makeFromTuple(data.backgroundColor);
            return settingsData;
        }
    }


    class SettingsDialog extends objects.Dialog {
        constructor() {
            super("settings-dialog");
            this.setTitle("Settings");
            this.settingsData = new SettingsData();

            this.fovYInput = new objects.NumberInput($("#fovY"), "fovY");
            this.renderPatienceInput = new objects.NumberInput($("#renderPatience"), "Render Patience");
            this.renderQualityInput = new objects.InputWithSlider($("#renderQuality"), "Render Quality");
            this.fieldPresentationJQ = this.nodeJQ.find("input[name=field-values-mode]");
            this.backgroundColorInput = new objects.ColorPicker($("#background-color-input"));

            this.init();
            this.hide();
        }
        init() {
            this.fovYInput.setDefaultValue(30);
            this.renderPatienceInput.setDefaultValue(40);
            this.renderQualityInput.setDefaultValue(10);
            this.renderQualityInput.setRange(new objects.Range(0.1, 10, 99));

            this.fieldPresMode = this.getFieldPresentationMode();

            let timeStepContnr = this.nodeJQ.find(".number-input[name=animation-time-step]");
            let placeholder = "";
            this.timeStepInput = new objects.NumberInput(timeStepContnr, placeholder);
            this.timeStepInput.setDefaultValue(500);
            this.timeStepInput.reset();
        }
        setSettingsData(data) {
            this.settingsData = data;
            this.fovYInput.setValue(data.fovY);
            this.renderPatienceInput.setValue(data.renderPatience);
            this.renderQualityInput.setValue(data.renderQuality);
            this.timeStepInput.setValue(data.animationInterval);
            this.setFieldPresentationMode(data.fieldPresentationMode);
            this.backgroundColorInput.setColor(data.backgroundColor);
        }
        getSettingsData() {
            this.settingsData.fovY = this.fovYInput.getValue();
            this.settingsData.fieldPresentationMode = this.getFieldPresentationMode();
            this.settingsData.renderPatience = this.renderPatienceInput.getValue();
            this.settingsData.renderQuality = this.renderQualityInput.getValue();
            this.settingsData.animationInterval = this.timeStepInput.getValue();
            this.settingsData.backgroundColor = this.backgroundColorInput.rgb;
            return this.settingsData;
        }
        cancel() {
            this.setFieldPresentationMode(this.fieldPresMode);
            
            this.hide();
            this.reset();
        }
        reset() {
            this.fovYInput.reset();
            this.renderPatienceInput.reset();
            this.renderQualityInput.reset();
            this.timeStepInput.reset();
        }
        getFieldPresentationMode() {
            return this.fieldPresentationJQ.filter(":checked").val();
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationJQ.each( (index, radioDom)=>{
                $(radioDom).prop('checked', $(radioDom).val() == mode);
            });
        }
    }


    class SettingsService {
        constructor() {
            this.dialog = new SettingsDialog();
            this.settingsData = new SettingsData();
            this.btnJQ = $("#settings-btn");

            this.btnJQ.click(this, e=>{
                e.data.btnJQ.blur();
                e.data.raiseDialog();
            });
            this.stop();
        }
        run() {
            this.restore();
            this.btnJQ.attr("disabled", false);
        }
        stop() {
            this.btnJQ.attr("disabled", true);
            this.dialog.cancel();
        }
        store() {
            let data = this.settingsData.getData();
            localStorage.setItem("settings", JSON.stringify(data));
        }
        restore() {
            let dataStr = localStorage.getItem("settings");
            if(dataStr == null) {
                this.notifyOther();
                return;
            }
            let data = JSON.parse(dataStr);
            this.settingsData = SettingsData.make(data);
            objects.ctfService.setBackgroundColor(
                this.settingsData.backgroundColor
            );
            this.sendData();
            this.notifyOther();
        }
        raiseDialog() {
            this.dialog.show();
            this.dialog.setSettingsData(this.settingsData);
            let handler = ()=>{
                this.settingsData = this.dialog.getSettingsData();
                this.sendData();
                this.notifyOther();
                this.store();
                objects.ctfService.setBackgroundColor(
                    this.settingsData.backgroundColor
                );
            };
            this.dialog.addSubmitHandler(handler.bind(this));
        }
        sendData() {
            let async = true;
            let fovY = this.settingsData.fovY;
            new objects.InputSetRequest("fovY", fovY).perform(async);
            let renderQuality = this.settingsData.renderQuality;
            new objects.InputSetRequest("renderQuality", renderQuality).perform(async);
            let renderPatience = this.settingsData.renderPatience;
            new objects.InputSetRequest("renderPatience", renderPatience).perform(async);

            let backgroundColor = this.settingsData.backgroundColor;
            new objects.InputSetRequest("backgroundColor", [
                backgroundColor.r / 255,
                backgroundColor.g / 255,
                backgroundColor.b / 255
            ]).perform(async);
        }
        notifyOther() {
            objects.timeService.setTimeStep(this.settingsData.animationInterval);
            
            let mode = this.settingsData.fieldPresentationMode;
            objects.fieldModeService.setFieldPresentationMode(mode);
        }
        hideDialog() {
            this.dialog.cancel();
        }
    }
    objects.settingsService = new SettingsService();
})()
