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
    class Req {
        constructor(url, paramsStr) {
            this.paramsStr = paramsStr;
            this.url = url;
            this.errorStr = `Error occured during the set request with. \n url: ${url}, params:${paramsStr}`;
        }
    }

    class SetRequest extends Req {
        constructor(url, paramsStr) {
            super(url, paramsStr);
        }
        perform(async = false) {
            let ajaxObj = {
                type:   "GET",
                url:    this.url,
                data:   this.paramsStr,
                async:  async
            };
            if(async) {
                $.ajax(ajaxObj).catch((jqXHR, textStatus)=>{
                    let errorMsg;
                    if(jqXHR.readyState == 0)
                        errorMsg = "Failed to connect with server. The outgoing request is unsent";
                    else
                        errorMsg = `Error occured during the request performance: ${textStatus}`;
                    objects.showError(errorMsg);
                });
            } else {
                ajaxObj.error = (jqXHR, textStatus)=> {
                    if(jqXHR.readyState == 0)
                        throw new Error("Failed to connect with server. The outgoing request is unsent");
                    else
                        throw new Error(`Error occured during the request performance: ${textStatus}`);
                };
                $.ajax(ajaxObj);
            }
        }
    }
    objects.SetRequest = SetRequest;


    class GetRequest extends Req {
        constructor(url, paramsStr) {
            super(url, paramsStr)
            this.received = null;
        }
        perform(async = false) {
            let ajaxObj = {
                type:       "GET",
                url:        this.url,
                data:       this.paramsStr,
                success:    received =>  {
                    this.received = JSON.parse(received)
                },
                async:      async
            };
            if(async) {
                $.ajax(ajaxObj).catch((jqXHR, textStatus)=>{
                    let errorMsg;
                    if(jqXHR.readyState == 0)
                        errorMsg = "Failed to connect with server. The outgoing request is unsent";
                    else
                        errorMsg = `Error occured during the request performance: ${textStatus}`;
                    objects.showError(errorMsg);
                });
            } else {
                ajaxObj.error = (jqXHR, textStatus)=> {
                    if(jqXHR.readyState == 0)
                        throw new Error("Failed to connect with server. The outgoing request is unsent");
                    else
                        throw new Error(`Error occured during the request performance: ${textStatus}`);
                };
                $.ajax(ajaxObj);
            }
            return this;
        }
        receivedData() {
            return this.received;
        }
    }
    objects.GetRequest = GetRequest;


    class InputSetRequest extends SetRequest {
        constructor(root, value) {
            let params = new URLSearchParams();
            params.append("root", root);
            if( value instanceof String )
                params.append( "value", value );
            else 
                params.append( "value", JSON.stringify(value) );
            super("vs/input/set", params.toString());
        }
    }
    objects.InputSetRequest = InputSetRequest;


    class InputGetRequest extends GetRequest {
        constructor(root) {
            let params = new URLSearchParams();
            params.append("root", root);
            super("vs/input/get", params.toString());
        }
    }
    objects.InputGetRequest = InputGetRequest;
})()