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
    class CtfService {
        constructor() {
            this.btnJQ = $("#ctf-dialog-raise-btn");
            this.ctfDialog = new objects.Ctf.Dialog();
            this.ctfDialog.setTitle("Color Transfer Function");

            this.ctf = new objects.Ctf.Func();
            this.ctf.setDefault();

            this.btnJQ.click(this.raiseDialog.bind(this));

            this.stop();
        }
        run() {
            this.restore();
            this.btnJQ.attr("disabled", false);
        }
        stop() {
            this.ctfDialog.cancel();
            this.btnJQ.attr("disabled", true);
        }
        restore() {
            let dataStr = localStorage.getItem("ctf");
            if(dataStr == null)
                return;
            let data = JSON.parse(dataStr);
            this.ctf = objects.Ctf.Func.restore(data);
            this.sendData();
        }
        store() {
            let data = this.ctf.makeDataToStore();
            localStorage.setItem("ctf", JSON.stringify(data));
        }
        sendData() {
            let async = true;
            let data = this.ctf.makeDataToSend();
            new objects.InputSetRequest("fieldParam.colorTransferFunction", data).perform(async);
        }
        raiseDialog() {
            this.ctfDialog.setCtf(this.ctf);
            this.ctfDialog.show();
            let handler = ()=> {
                this.ctf = this.ctfDialog.getCtf();
                this.onUpdate();
            };
            this.ctfDialog.addSubmitHandler(handler.bind(this));
        }
        onUpdate() {
            this.store();
            this.sendData();
        }
        setBackgroundColor(rgb) {
            this.ctfDialog.setBackgroundColor(rgb);
        }
    }
    objects.ctfService = new CtfService();
})()