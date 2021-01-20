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
    let InpSetReq = objects.InputSetRequest;

    class SingleIsosurfaceUi extends objects.Element {
        constructor() {
            super("single-isosurface-control");
            let containerJQ = this.nodeJQ.find("div[name=primaryFieldValue]");
            this.fieldLevelInp = new objects.InputWithSlider(containerJQ, "Field level");
            this.fieldLevelInp.addEventListener("change", ()=>{
                this.dispatchEvent(new Event("change"));
            });
            
            this.fieldRange = new objects.Range(0, 1, 100);
            this.fieldLevelInp.setRange(this.fieldRange);
            this.fieldPresentationMode = "absolute";
        }
        setData(data) {
            if(data != null)
                this.fieldLevelInp.setValue(data);
        }
        getData() {
            return this.fieldLevelInp.isCompleted()? this.fieldLevelInp.getValue() : null;
        }
        disable() {
            this.fieldLevelInp.disable();
        }
        enable() {
            this.fieldLevelInp.enable();
        }
        reset() {
            this.fieldLevelInp.reset();
        }
        setFieldLevelInp(level) {
            if(isNaN(level))
                throw new Error("Tried to set a nan for level of isosurface");
            this.fieldLevelInp.setValue(level);
        }
        setFieldValuesRange(range) {
            this.fieldRange = range;
            this.updateFieldInputsRange();
        }
        extractLevels() {
            return [this.fieldLevelInp.getRelativeValue()];
        }
        isCompleted() {
            return this.fieldLevelInp.isCompleted();
        }
        updateFieldInputsRange() {
            switch (this.fieldPresentationMode) {
                case "absolute":
                {
                    this.fieldLevelInp.setRange(this.fieldRange);
                    let defaultVal = (this.fieldRange.min + this.fieldRange.max) / 2;
                    this.fieldLevelInp.setDefaultValue(defaultVal);
                    break;
                }
                case "relative":
                {
                    this.fieldLevelInp.setRange(new objects.Range(0, 1, 100));
                    this.fieldLevelInp.setDefaultValue(0.5);
                    break;
                }
            }
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationMode = mode;
            this.updateFieldInputsRange();
        }
    }

    class LevelInput extends EventTarget {
        constructor() {
            super();
            this.nodeJQ = $("<div>").addClass("isosurface-level");
            this.input = new objects.InputWithSlider(this.nodeJQ, "Isosurface level");
            this.input.addEventListener("change", ()=>{
                this.dispatchEvent(new Event("change"));
            })
            
            this.fieldPresentationMode = "relative";
            this.fieldRange = new objects.Range(0, 1, 100);
        }
        focusin() {
            this.nodeJQ.addClass("focused");
        }
        focusout() {
            this.nodeJQ.removeClass("focused");
        }
        appendTo(nodeJQ) {
            this.nodeJQ.appendTo(nodeJQ);
        }
        getValue() {
            return this.input.getValue();
        }
        setValue(level) {
            this.input.setValue(level);
        }
        isEmpty() {
            return this.input.isEmpty();
        }
        disable() {
            this.nodeJQ.attr("disabled", true);
        }
        enable() {
            this.nodeJQ.attr("disabled", false);
        }
        setFieldRange(range) {
            this.fieldRange = range;
            this.updateInputRange();
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationMode = mode;
            this.updateInputRange();
        }
        getRelativeValue() {
            return this.input.getRelativeValue();
        }
        setRelativeValue(relValue) {
            this.input.setRelativeValue(relValue);
        }
        isCompleted() {
            return this.input.isCompleted();
        }

        // private
        updateInputRange() {
            if(this.fieldPresentationMode == "relative")
                this.input.setRange(new objects.Range(0, 1, 100));
            else
                this.input.setRange(this.fieldRange);
        }
    }

    class LevelsList extends EventTarget {
        constructor() {
            super();
            this.nodeJQ = $(`#field-values-table`);
            this.levelInputs = [];
            this.focusedLevel = null;
            
            this.fieldRange = new objects.Range(0, 1, 100);
            this.fieldPresentationMode = "absolute";
        }
        focusLevel(level) {
            if( this.focusedLevel )
                this.focusedLevel.focusout();
            level.focusin();
            this.focusedLevel = level;
        }
        addLevel(levelInput) {
            levelInput.setFieldPresentationMode(this.fieldPresentationMode);
            levelInput.setFieldRange(this.fieldRange);

            levelInput.nodeJQ.appendTo(this.nodeJQ);
            this.levelInputs.push(levelInput);

            levelInput.nodeJQ.click(this, (e)=>{
                this.findByDomAndFocus(e.currentTarget);
            });
            levelInput.addEventListener("change", ()=>{
                this.dispatchEvent(new Event("change"));
            });
        }
        createLevel() {
            let levelInput = new LevelInput();
            this.addLevel(levelInput);
        }
        removeFocused() {
            if(!this.focusedLevel)
                return;
            let index = this.levelInputs.findIndex( levelInput => {
                return this.focusedLevel == levelInput 
            });
            this.levelInputs.splice(index, 1);
            if( this.focusedLevel )
                this.focusedLevel.nodeJQ.remove();
            this.dispatchEvent(new Event("change"));
        }
        clear() {
            this.levelInputs.forEach( (levelInput)=> levelInput.nodeJQ.remove() );
            this.levelInputs = [];
            this.dispatchEvent(new Event("change"));
        }
        extractLevelValues() {
            let result = [];
            this.levelInputs.forEach( (levelInput)=>{
                if(levelInput.isEmpty())
                    return true;
                result.push(levelInput.getRelativeValue());
            });
            return result;
        }
        disable() {
            this.levelInputs.forEach( levelInput => levelInput.disable() );
        }
        enable() {
            this.levelInputs.forEach( levelInput => levelInput.enable() );
        }
        setAcceptableFieldRange(range) {
            this.fieldRange = range;
            this.levelInputs.forEach(levelInput => levelInput.setFieldRange(range));
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationMode = mode;
            this.levelInputs.forEach( levelInput =>
                levelInput.setFieldPresentationMode(mode)
            );
        }
        findByDom(DomObj) {
            let levelInput = this.levelInputs.find( (levelInput)=>{
                return levelInput.nodeJQ[0] == DomObj;
            });
            return levelInput;
        }
        findByDomAndFocus(DomObj) {
            let levelInput = this.findByDom(DomObj);
            if(levelInput)
                this.focusLevel(levelInput);
        }
    }

    class RangeLevelsDialog extends objects.Dialog {
        constructor() {
            super("range-dialog");
            this.setTitle("Add range of isosurface levels");

            this.beginInput = new objects.InputWithSlider($("#range-begin-input"), "Begin");
            this.beginInput.setDefaultValue(0);
            this.beginInput.setValue(0);

            this.endInput = new objects.InputWithSlider($("#range-end-input"), "End");
            this.endInput.setDefaultValue(1);
            this.endInput.setValue(1);

            this.nStepsInput = new objects.NumberInput($("#range-steps-input"), "Steps Number");
            this.nStepsInput.setDefaultValue(5);
            this.nStepsInput.setValue(5);

            this.fieldPresentationMode = "absolute";
            this.fieldRange = new objects.Range(0, 1, 100);
        }
        getRelativeLevelsArray() {
            let begin = this.beginInput.getRelativeValue();
            let end = this.endInput.getRelativeValue();
            let nSteps = this.nStepsInput.getValue();
            let range = new objects.Range(begin, end, nSteps);
            return range.getValuesArray();
        }
        reset() {
            this.beginInput.reset();
            this.endInput.reset();
            this.nStepsInput.reset();
        }
        setFieldRange(range) {
            this.fieldRange = range;
            this.updateFieldInputsRanges();
        }
        setFieldPresentationMode(mode) {
            this.fieldPresentationMode = mode;
            this.updateFieldInputsRanges();
        }
        updateFieldInputsRanges() {
            switch(this.fieldPresentationMode) {
                case "absolute":
                {
                    this.beginInput.setRange(this.fieldRange);
                    this.beginInput.setDefaultValue(this.fieldRange.min);
                    this.endInput.setRange(this.fieldRange);
                    this.endInput.setDefaultValue(this.fieldRange.max);
                    break;
                }
                case "relative":
                {
                    let range = new objects.Range(0, 1, 100);
                    this.beginInput.setRange(range);
                    this.beginInput.setDefaultValue(0);
                    this.beginInput.setValue(0);
                    this.endInput.setRange(range);
                    this.endInput.setDefaultValue(1);
                    this.endInput.setValue(1);
                    break;
                }
            }
        }
    }

    class MultipleIsosurfaceUi extends objects.Element{
        constructor() {
            super("multiple-fields-control");
            this.levelsList = new LevelsList();
            this.rangeDialog = new RangeLevelsDialog();

            this.addBtnJQ = this.nodeJQ.find("#add-isosurface-btn");
            this.removeBtnJQ = this.nodeJQ.find("#remove-isosurface-btn");
            this.rangeBtnJQ = this.nodeJQ.find("#range-isosurface-btn");
            this.clearBtnJQ = this.nodeJQ.find("#clear-isosurfaces-btn");

            this.addBtnJQ.click( this.createLevelInput.bind(this) );
            this.removeBtnJQ.click( this.removeFocusedLevel.bind(this) );
            this.rangeBtnJQ.click( this.rangeBtnClicked.bind(this) );
            this.clearBtnJQ.click( this.clearLevels.bind(this) );

            this.levelsList.addEventListener("change", ()=>{
                this.dispatchEvent(new Event("change"));
            });
        }
        setData(data) {
            this.clearLevels();
            data.forEach(level => {
                let li = new LevelInput();
                li.setValue(level);
                this.levelsList.addLevel(li);
            }, this);
        }
        getData() {
            return this.levelsList.extractLevelValues();
        }
        createLevelInput() {
            this.levelsList.createLevel();
        }
        removeFocusedLevel() {
            this.levelsList.removeFocused();
        }
        rangeBtnClicked() {
            let onSubmit = ()=>{
                let relLevels = this.rangeDialog.getRelativeLevelsArray();
                relLevels.forEach( (level)=>{
                    let levelInput = new LevelInput();
                    levelInput.setRelativeValue(level);
                    this.levelsList.addLevel(levelInput);
                });
            };
            this.rangeDialog.addSubmitHandler(onSubmit.bind(this));
            this.rangeDialog.addSubmitHandler(this.dispatchEvent.bind(this, new Event("change")));
            this.rangeDialog.show();
        }
        clearLevels() {
            this.levelsList.clear();
        }
        reset() {
            this.levelsList.clear();
            this.rangeDialog.cancel();
        }
        hide() {
            if( this.rangeDialog )
                this.rangeDialog.hide();
            super.hide();
        }
        disable() {
            this.levelsList.disable();
            this.addBtnJQ.attr("disabled", true);
            this.removeBtnJQ.attr("disabled", true);
            this.clearBtnJQ.attr("disabled", true);
            this.rangeBtnJQ.attr("disabled", true);
        }
        enable() {
            this.levelsList.enable;
            this.addBtnJQ.attr("disabled", false);
            this.removeBtnJQ.attr("disabled", false);
            this.clearBtnJQ.attr("disabled", false);
            this.rangeBtnJQ.attr("disabled", false);
        }
        extractLevels() {
            return this.levelsList.extractLevelValues();
        }
        setFieldValuesRange(range) {
            this.levelsList.setAcceptableFieldRange(range);
            this.rangeDialog.setFieldRange(range);
        }
        isCompleted() {
            let levels = this.extractLevels();
            if(levels.length == 0)
                return false;
            return true;
        }
        setFieldPresentationMode(mode) {
            this.levelsList.setFieldPresentationMode(mode);
            this.rangeDialog.setFieldPresentationMode(mode);
        }
    }

    class IsosurfaceUi extends objects.Element {
        constructor() {
            super("isosurface-control");
            this.isosurfacesNumberSelectJQ = $("#isosurfaces-number");

            this.singleIsosurfaceUi = new SingleIsosurfaceUi();
            this.multipleIsosurfaceUi = new MultipleIsosurfaceUi();
            this.activeUi = null;

            let opacityJQ = $("#isosurface-control div[name=opacity]");
            this.opacityInput = new objects.InputWithSlider(opacityJQ, "Opacity");
            this.opacityInput.setRange( new objects.Range(0,1,100) );
            this.opacityInput.setDefaultValue(1);
            this.opacityInput.reset();

            this.secondaryFieldSelectJQ = $("#secondary-field-select");
            this.okBtnJQ = this.nodeJQ.find("button[value=Ok]");

            this.okBtnJQ.click( this.submit.bind(this) );
            this.defineActiveUi();

            let changeCb = ()=>{
                if(!objects.fieldModeService.isRestoring)
                    this.store();
                if(this.isCompleted())
                    this.submit();
                else
                    objects.readiness.fieldMode = false;
            };
            this.isosurfacesNumberSelectJQ.change(this, e=>{
                e.data.defineActiveUi();
                changeCb.apply(e.data);
            });
            this.singleIsosurfaceUi.addEventListener("change", changeCb.bind(this));
            this.multipleIsosurfaceUi.addEventListener("change", changeCb.bind(this));
            this.isosurfacesNumberSelectJQ.change( changeCb.bind(this) );
            this.opacityInput.addEventListener("change", changeCb.bind(this));
            this.secondaryFieldSelectJQ.change( changeCb.bind(this) );
            
            this.hide();
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
        show() {
            super.show();
            this.defineActiveUi();
            if(this.isCompleted())
                this.submit();
            else
                objects.readiness.fieldMode = false;

        }
        defineActiveUi() {
            let mode = this.isosurfacesNumberSelectJQ.val();
            if(mode == "single") {
                if(!this.isHidden) {
                    this.singleIsosurfaceUi.show();
                    this.multipleIsosurfaceUi.hide();
                }
                this.activeUi = this.singleIsosurfaceUi;
            }
            else {
                if(!this.isHidden) {
                    this.multipleIsosurfaceUi.show();
                    this.singleIsosurfaceUi.hide();
                }
                this.activeUi = this.multipleIsosurfaceUi;
            }
        }
        restore() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            
            let dataStr = localStorage.getItem(problem);
            if(dataStr == null)
                return;
            let data = JSON.parse(dataStr);
            let fieldModeData = data.fieldModeData;
            if(!fieldModeData)
                return;

            if(!isNaN(fieldModeData.opacity))
                this.opacityInput.setValue(fieldModeData.opacity);
            
            this.secondaryFieldSelectJQ.val(fieldModeData.secondaryField);

            this.isosurfacesNumberSelectJQ.val(fieldModeData.mode);
            this.isosurfacesNumberSelectJQ.change();

            this.singleIsosurfaceUi.setData(fieldModeData.singleData);
            this.multipleIsosurfaceUi.setData(fieldModeData.multipleData);
            if(!this.isHidden)
                this.submit();
        }
        store() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;
            
            let dataStr = localStorage.getItem(problem);
            let data = dataStr==null? {} : JSON.parse(dataStr);

            let fieldModeData = {};
            fieldModeData.opacity = this.opacityInput.getRelativeValue();
            fieldModeData.secondaryField = this.secondaryFieldSelectJQ.val();
            fieldModeData.mode = this.isosurfacesNumberSelectJQ.val();
            fieldModeData.singleData = this.singleIsosurfaceUi.getData();
            fieldModeData.multipleData = this.multipleIsosurfaceUi.getData();

            data.fieldModeData = fieldModeData;
            localStorage.setItem(problem, JSON.stringify(data));
        }
        update() {
            this.sendData();
            if(!objects.fieldModeService.isRestoring)
                this.store();
        }
        sendData() {
            let levels = this.activeUi.extractLevels();
            if(levels.length == 0)
                throw new Error("No field defined");
            let secondaryField = this.secondaryFieldSelectJQ.val();
            if(secondaryField == "none")
                secondaryField = null;
            if(levels.length == 1) {
                if(secondaryField) {
                    new InpSetReq("fieldMode", "ValueOnIsosurface").perform();
                    new InpSetReq("fieldParam.isosurfaceLevel", levels[0]).perform();
                    new InpSetReq("fieldParam.secondaryField", secondaryField).perform();
                } else {
                    new InpSetReq("fieldMode", "Isosurface").perform();
                    new InpSetReq("fieldParam.isosurfaceLevel", levels[0]).perform();
                }
            } else {
                if(secondaryField) {
                    new InpSetReq("fieldMode", "ValueOnIsosurfaces").perform();
                    new InpSetReq("fieldParam.isosurfaceLevels", levels).perform();
                    new InpSetReq("fieldParam.secondaryField", secondaryField).perform();
                } else {
                    new InpSetReq("fieldMode", "Isosurfaces").perform();
                    new InpSetReq("fieldParam.isosurfaceLevels", levels).perform();
                }
            }
            new InpSetReq("fieldParam.isosurfaceOpacity", this.opacityInput.getValue()).perform();
        }
        hideDialog() {
            this.multipleIsosurfaceUi.rangeDialog.cancel();
        }
        disable() {
            this.isosurfacesNumberSelectJQ.attr("disabled", true);
            this.singleIsosurfaceUi.disable();
            this.multipleIsosurfaceUi.disable();
            this.opacityInput.disable();
            this.secondaryFieldSelectJQ.attr("disabled", true);
        }
        enable() {
            this.isosurfacesNumberSelectJQ.attr("disabled", false);
            this.singleIsosurfaceUi.enable();
            this.multipleIsosurfaceUi.enable();
            this.opacityInput.enable();
            this.secondaryFieldSelectJQ.attr("disabled", false);
        }
        setFieldNames(fieldNames) {
            this.secondaryFieldSelectJQ.find("option").each( (index, optionDom)=>{
                let optionVal = $(optionDom).val();
                if(optionVal == "none")
                    return true;
                $(optionDom).remove();
            });
            fieldNames.forEach( (field)=>{
                let optionJQ = $("<option>")
                    .attr("value", field)
                    .text(field);
                optionJQ.appendTo(this.secondaryFieldSelectJQ);
            }, this);
            this.secondaryFieldSelectJQ.val("none");
        }
        setPrimaryField(primaryField) {
            this.secondaryFieldSelectJQ.find("option[disabled]").attr("disabled", false);
            if(primaryField == "none") {
                this.secondaryFieldSelectJQ.attr("disabled", true);
                this.secondaryFieldSelectJQ.val("none");
                return;
            }
            else  {
                this.secondaryFieldSelectJQ.attr("disabled", false);
                this.secondaryFieldSelectJQ
                    .find(`option[value=${primaryField}]`)
                    .attr("disabled", true);
            }
            let secField = this.secondaryFieldSelectJQ[0].value;
            if(secField == primaryField)
                this.secondaryFieldSelectJQ.val("none");
        }
        setFieldRange(range) {
            this.singleIsosurfaceUi.setFieldValuesRange(range);
            this.multipleIsosurfaceUi.setFieldValuesRange(range);
        }
        reset() {
            this.singleIsosurfaceUi.fieldLevelInp.reset();
            this.opacityInput.reset();
            this.secondaryFieldSelectJQ.val("none");
            this.defineActiveUi();
        }
        isCompleted() {
            if(!this.activeUi.isCompleted())
                return false;
            if(!this.opacityInput.isCompleted())
                return false;
            return true;
        }
        setFieldPresentationMode(mode) {
            this.singleIsosurfaceUi.setFieldPresentationMode(mode);
            this.multipleIsosurfaceUi.setFieldPresentationMode(mode);
        }
    }
    objects.isosurfaceUi = new IsosurfaceUi();
})()