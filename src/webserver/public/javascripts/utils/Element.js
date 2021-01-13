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
    class Element extends EventTarget {
        constructor( id ) {
            super();
            this.nodeJQ = $(`#${id}`);
            this.isHidden = true;
        }

        show() {
            this.nodeJQ[0].dispatchEvent( new Event("visible") );
            
            this.nodeJQ.addClass("visible");
            this.nodeJQ.removeClass("hidden");
            this.nodeJQ.find('*').each( (index, elementDOM)=>{
                $(elementDOM).removeClass("hidden");
                $(elementDOM).addClass("visible");
            });
            this.isHidden = false;
        }

        hide() {
            this.nodeJQ.addClass("hidden");
            this.nodeJQ.removeClass("visible");
            this.nodeJQ[0].dispatchEvent( new Event("hidden") );
            this.nodeJQ.find('*').each( (index, elementDOM)=>{
                $(elementDOM).addClass("hidden");
                $(elementDOM).removeClass("visible");
            });
            this.isHidden = true;
        }

        reset() {
            let inputs = this.nodeJQ.find("input");
            for( let i = 0; i < inputs.length; ++i ) {
                switch( inputs[i].type ) {
                    case 'text':
                        inputs[i].value = '';
                        break;
                    case 'checkbox':
                        inputs[i].checked = false;
                        break;
                    case 'number':
                        inputs[i].value = '';
                        break;
                    default:
                        break;
                }
            }
        }
    }
    objects.Element = Element;
})()