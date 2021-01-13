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
    class ViewportSizeService {
        constructor() {
            this.viewportJQ = $("canvas");
            
            $(window).resize( this.sendViewportSize.bind(this) );
            this.isRun = false;
            this.stop();
        }
        run() {
            this.isRun = true;
            this.sendViewportSize();
        }
        stop() {
            this.isRun = false;
        }
        sendViewportSize() {
            try {
                if(!this.isRun)
                    return;
                let w = this.viewportJQ.width();
                let h = this.viewportJQ.height();
                new objects.InputSetRequest( "viewportSize", [w,h] ).perform();
            } catch (error) {
                objects.showError(error.message);
            }
        }
    }
    objects.viewportSizeService = new ViewportSizeService();
})()