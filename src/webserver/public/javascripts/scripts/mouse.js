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
    let canvasJQ = $("#viewport>canvas");
    let trackMouse = false;

    let mouseState = {
        x:          0,
        y:          0,
        wheelDelta: 0,
        f:          0,
        in:         false,
        anyPressed: ()=> mouseState.f != 0
    };
    canvasJQ.mouseenter( ()=> {
        mouseState.in = true;
    });
    canvasJQ.mouseout( ()=> {
        mouseState.in = false;
    });
    
    function performReq() {
        if ( $("#problem-select").val() == "none" )
            return;
        let params = new URLSearchParams();
        params.append("x", mouseState.x);
        params.append("y", mouseState.y);
        params.append("w", mouseState.wheelDelta);
        params.append("f", mouseState.f);
        new objects.SetRequest(
            "vs/mouse",
            params.toString()
        ).perform(true);
    }

    function setFlag( flag, valid ) {
        if( valid )
            mouseState.f |= flag;
        else 
            mouseState.f &= (~flag);
    }

    $(document).keydown( (e)=>{
        if( !mouseState.anyPressed() && mouseState.in )
            trackMouse = true;
        if( !(mouseState.in || trackMouse) )
            return;
        switch(e.key) {
            case "Control":
                setFlag(0x10, true);
                break;
            case "Shift":
                setFlag(0x08, true);
                break;
            case "Alt":
                setFlag(0x20, true);
                break;
            default:
                break;
        }
        performReq();
    });

    $(document).keyup( (e)=>{
        if( !trackMouse )
            return;
        switch (e.key) {
            case "Control":
                setFlag(0x10, false);
                break;
            case "Shift":
                setFlag(0x08, false);
                break;
            case "Alt": 
                setFlag(0x20, false);
                break;
            default:
                break;
        }
        if(!mouseState.anyPressed())
            trackMouse = false;
    });

    $(document).mousedown( (e)=>{
        if( !mouseState.anyPressed() && mouseState.in )
            trackMouse = true;
        if(!trackMouse)
            return;
        switch (e.button) {
            case 0:     // LMB 
                setFlag(0x01, true);
                break;
            case 1:     // Wheel
                setFlag(0x04, true);
                break;
            case 2:     // RMB
                setFlag(0x02, true);
                break;
            default:    
                break;
        }
        performReq();
    });

    $(document).mouseup( (e)=>{
        if( !trackMouse )
            return;
        switch (e.button) {
            case 0:     // LMB
                setFlag(0x01, false);
                break;
            case 1:     // Wheel
                setFlag(0x04, false); 
                break;
            case 2:     // RMB
                setFlag(0x02, false);
            default:
                break;
        }
        if(!mouseState.anyPressed())
            trackMouse = false;
        performReq();
    });

    let ts = Date.now();
    $(document).mousemove(canvasJQ, (e)=> {
        let ts2 = Date.now();
        if(ts2 - ts < 100)
            return;
        let cPos = e.data.position();
        let x = e.clientX - cPos.left;
        let y = e.clientY - cPos.top;
        mouseState.x = Math.round(x);
        mouseState.y = Math.round(y);
        ts = ts2;
        if(trackMouse)
            performReq();
    });


    function mousewheel(delta) {
        mouseState.wheelDelta += delta;
        performReq();
    }
    // chrome
    canvasJQ.bind('mousewheel', (e)=>{
        if(!mouseState.in)
            return;
        let ratio = 3 / 53; // necessary to make wheel delta in chrome equal delta in firefox 
        mousewheel( e.originalEvent.deltaY * ratio );
    });
    // ff
    canvasJQ.bind("DOMMouseScroll", (e)=>{
        if(!mouseState.in)
            return;
        mousewheel( e.detail )
    });

    canvasJQ.on("contextmenu", false);
    $(window).blur( () =>{
        if(!trackMouse)
            return;
        mouseState.f = 0;
        trackMouse = false;
        performReq();
    });
})();