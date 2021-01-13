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
    class WithThresholdUi extends objects.Element {
        constructor(fieldMode, id) {
            super(id);
            this.fieldMode = fieldMode;

            let inputDivJQ = this.nodeJQ.find("div[name=threshold]");
            this.thresholdInput = new objects.InputWithSlider(inputDivJQ, "Threshold");
            this.okBtnJQ = this.nodeJQ.find("button[value=Ok]");
            
            this.okBtnJQ.click( this.submit.bind(this) );
            this.thresholdInput.addEventListener("change", ()=>{
                if(this.thresholdInput.isCompleted())
                    this.submit();
                else
                    objects.readiness.fieldMode = false;
            });

            this.fieldRange = new objects.Range(0, 1, 100);
            this.fieldPresentationMode = "relative";
            
            this.hide();
        }
        update() {
            new objects.InputSetRequest("fieldMode", this.fieldMode).perform();
            let threshold = this.thresholdInput.getRelativeValue();
            new objects.InputSetRequest("fieldParam.threshold", threshold).perform();

            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            let data = dataStr==null? {} : JSON.parse(dataStr);
            data[this.fieldMode] = threshold;
            localStorage.setItem(problem, JSON.stringify(data));
        }
        restore() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            let dataStr = localStorage.getItem(problem);
            if(dataStr == null)
                return;
            let data = JSON.parse(dataStr);
            let threshold = data[this.fieldMode];
            if(!isNaN(threshold)) {
                this.thresholdInput.setValue(threshold);
                if(this.thresholdInput.isCompleted() && !this.isHidden)
                    this.submit();
                else
                    objects.readiness.fieldMode = false;
            }
        }
        submit() {
            try {
                this.update();
                objects.readiness.fieldMode = true;
            } catch (error) {
                objects.readiness.fieldMode = false;
                objects.showError(error.message);
            }
        }
        cancel() {
            this.reset();
        }
        disable() {
            this.nodeJQ.find("button,input,select").attr("disabled", true);
        }
        enable() {
            this.nodeJQ.find("button,input,select").attr("disabled", false);
        }
        setAcceptableInputValuesRange(range) {
            this.fieldRange = range;
            this.updateInputRange();
        }
        updateInputRange() {
            switch (this.fieldPresentationMode) {
                case "absolute":
                {
                    this.thresholdInput.setRange(this.fieldRange);
                    let min = this.fieldRange.min;
                    let max = this.fieldRange.max;
                    let middle = (min + max) / 2;
                    this.thresholdInput.setDefaultValue(middle);
                    break;
                }
                case "relative":
                {
                    this.thresholdInput.setRange(new objects.Range(0, 1, 100));
                    this.thresholdInput.setDefaultValue(0.5);
                    break;
                }
                default:
                    break;
            }
        }
        show() {
            super.show();
            if(this.thresholdInput.isCompleted())
                this.submit();
            else
                objects.readiness.fieldMode = false;
        }
        reset() {
            this.thresholdInput.reset();
        }
        isCompleted() {
            return this.thresholdInput.isCompleted();
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationMode = mode;
            this.updateInputRange();
        }
    }
    objects.mipUi = new WithThresholdUi("MaxIntensityProjection", "MIP-control");
    objects.domainVoxelsUi = new WithThresholdUi("DomainVoxels", "domain-voxels");
    objects.argbLightUi = new WithThresholdUi("ArgbLight", "argb-light-control");
    objects.argbUi = new WithThresholdUi("Argb", "argb-control");
})()
