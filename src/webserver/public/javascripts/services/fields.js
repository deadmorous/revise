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
    class FieldsService {
        constructor() {
            this.fieldsInfo = [];
            
            this.primaryFieldSelectJQ = $("#primary-field-select");
            this.primaryField = null;
            
            this.primaryFieldSelectJQ.change( this.primaryFieldChanged.bind(this) );
            this.stop();
        }
        run() {
            this.updateFields();
            this.primaryFieldSelectJQ.attr("disabled", false);
            this.restorePrimaryField();
        }
        stop() {
            this.primaryFieldSelectJQ.attr("disabled", true);
        }
        restorePrimaryField() {
            let problem = $("#problem-select").val();
            if(problem == "none") 
                return;
            let data = localStorage.getItem(problem);
            if(data == null)
                return;
            data = JSON.parse(data);
            if(data.primaryField) {
                this.primaryFieldSelectJQ.val(data.primaryField);
                this.primaryFieldSelectJQ.change();
            }
        }
        storePrimaryField() {
            let problem = $("#problem-select").val();
            if(problem == "none") 
                return;
            let dataStr = localStorage.getItem(problem);
            let data = {};
            if(dataStr != null)
                data = JSON.parse(dataStr);
            let primaryField = this.primaryFieldSelectJQ.val();
            if(primaryField != "none")
                data.primaryField = primaryField;
            else
                delete data.primaryField;
            localStorage.setItem(problem, JSON.stringify(data));
        }
        primaryFieldChanged() {
            try {
                this.storePrimaryField();
                let primaryFieldVal = this.primaryFieldSelectJQ.val();
                objects.isosurfaceUi.setPrimaryField(primaryFieldVal);
                this.primaryField = primaryFieldVal=="none"? null : primaryFieldVal;
                if(this.primaryField) {
                    new objects.InputSetRequest("primaryField", this.primaryField).perform();
                    
                    let fieldInfo = this.fieldsInfo.find(fieldInfo => {
                        return fieldInfo.name == this.primaryField;
                    }, this);

                    objects.fieldModeService.setPrimaryField(fieldInfo);
                    objects.readiness.primaryField = true;
                } else 
                    objects.readiness.primaryField = false;
            } catch(error) {
                objects.readiness.primaryField = false;
                objects.showError(error.message);
            }
        }
        updateFields() {
            let req = new objects.GetRequest("vs/fields/names","");
            let fields = req.perform().receivedData();
            objects.isosurfaceUi.setFieldNames(fields);
            this.fieldsInfo = fields.map(field => {
                let paramsStr = new URLSearchParams({f: field}).toString();
                let req = new objects.GetRequest("vs/fields/range", paramsStr);
                let range = req.perform().receivedData();
                return {
                    name: field,
                    range: range
                };
            });
            
            // update field options in primary field select: remove old and insert new options
            this.primaryFieldSelectJQ.find("option").each( (index, optionDom) => {
                $(optionDom).remove();
            })
            fields.forEach( (fieldName)=>{
                $("<option>")
                    .attr("value", fieldName)
                    .text(fieldName)
                    .appendTo(this.primaryFieldSelectJQ);
            }, this);
            this.primaryFieldSelectJQ.val(fields[0]);
            this.primaryFieldSelectJQ.change();
        }
    }
    objects.fieldsService = new FieldsService();
})()