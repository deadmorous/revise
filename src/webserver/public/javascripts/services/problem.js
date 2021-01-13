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
    class ProblemService extends EventTarget {
        constructor(){
            try {
                super();

                this.currentProblem = null;
                this.selectJQ = $("#problem-select");
                
                this.receiveProblems();
                this.defineSelectOptions();
                this.selectJQ.change( this.problemChanged.bind(this) );
            } catch (error) {
                objects.showError(error.message);
            }
        }
        run() {
            this.selectJQ.attr("disabled", false);
        }
        stop() {
            this.selectJQ.val("none");
            this.problemChanged();
            this.selectJQ.attr("disabled", true);
        }
        restore() {
            let currentProblem = localStorage.getItem("currentProblem");
            if(currentProblem != null)
                this.selectJQ.val(currentProblem);
            else 
                this.selectJQ.val("none");
            this.selectJQ.change();
        }
        store() {
            let currentProblem = this.selectJQ.val();
            if(currentProblem != "none")
                localStorage.setItem("currentProblem", currentProblem);
            else
                localStorage.removeItem("currentProblem");
        }
        problemChanged() {
            try {
                this.store();
                this.defineCurrentProblem();
                this.sendCurrentProblemPath();
                if(this.currentProblem) {
                    objects.settingsService.run();
                    objects.cameraTransformService.run();
                    objects.viewportSizeService.run();
                    objects.clippingPlanesService.run();
                    objects.fieldsService.run();
                    objects.fieldModeService.run();
                    objects.timeService.run();
                    objects.ctfService.run();
                    
                    objects.readiness.problem = true;
                } else {
                    objects.cameraTransformService.stop();
                    objects.viewportSizeService.stop();
                    objects.clippingPlanesService.stop();
                    objects.fieldsService.stop();
                    objects.fieldModeService.stop();
                    objects.timeService.stop();
                    objects.settingsService.stop();
                    objects.ctfService.stop();

                    objects.readiness.problem = false;
                }
                this.dispatchEvent(new Event("changed"));
            } catch (error) {
                objects.showError(error.message);
                objects.readiness.problem = false;
            }
        }
        receiveProblems() {
            let req = new objects.GetRequest("vs/problems", "").perform();
            this.problems = req.receivedData();
        }
        defineSelectOptions() {
            this.problems.forEach((problem) => {
                let inserted = $("<option>")
                    .attr( "value", problem.HTMLOptionValue )
                    .text( problem.name );
                inserted.appendTo( this.selectJQ );
            }, this);
        }
        defineCurrentProblem() {
            let optionVal = this.selectJQ.val();
            this.currentProblem = this.problems.find((problem)=>{
                return problem.HTMLOptionValue == optionVal;
            });
        }
        sendCurrentProblemPath() {
            if(this.currentProblem) {
                let path = this.currentProblem.path;
                new objects.InputSetRequest("problemPath", path).perform();
            }
        }
    }
    objects.problemService = new ProblemService();
})()