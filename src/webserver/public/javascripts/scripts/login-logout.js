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
    let logBtnJQ = $("#login-logout");
    let isLogedin = true;
    logBtnJQ.click( ()=>{
        try {
            if(isLogedin) {
                new objects.SetRequest("vs/logout", "")
                    .perform();
                objects.problemService.stop();
                logBtnJQ
                    .removeClass("logout")
                    .addClass("login");
                isLogedin = false;
            } else {
                new objects.SetRequest("vs/logout", "")
                    .perform();
                objects.problemService.run();
                logBtnJQ
                    .removeClass("login")
                    .addClass("logout");
                isLogedin = true;
            }
        } catch(err) {
            objects.showError(err);
        }
    });


    let req = new objects.GetRequest("vs/alive/get");
    let delay = req.perform().receivedData();
    function sendAlive() {
        if(!isLogedin) 
            return;
        let async = true;
        new objects.SetRequest("vs/alive/update", "").perform(async);
    }
    setInterval(sendAlive, delay);
})()