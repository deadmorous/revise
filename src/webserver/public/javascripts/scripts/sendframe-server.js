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
    let req = new objects.GetRequest("vs/sourceId", "").perform();
    objects.sourceId = req.receivedData();

    let canvasDOM = $("#viewport>canvas")[0];
    let cxt = canvasDOM.getContext('2d');

    let wsUri = "ws://" + document.domain + ":1234";
    let websocket = null;

    function drawImage(src) {
        let img = new Image();
        img.src = src;
        img.onload = ()=>{
            canvasDOM.width = img.naturalWidth;
            canvasDOM.height = img.naturalHeight;
            cxt.drawImage(img, 0, 0);
        };
    }

    let measOutputJQ = $("#measureOutput");
    function outputMeasurements(numbers) {
        let add = "";
        add += `Shared mem frame number: ${numbers[0]}\n`;
        add += `Frame level: ${numbers[2]}\n`;
        add += `Rendering duration: ${numbers[3]}\n`;
        add += `Client frame number: ${numbers[1]}\n`;
        add += `Frame quality: ${numbers[4]}\n`;
        add += '\n';
        
        let out = measOutputJQ.val();
        out += add;
        measOutputJQ.val(out);
        measOutputJQ[0].scrollTop = measOutputJQ[0].scrollHeight;
    }

    function sendToServer(numbers) {
        let async = true;
        let params = new URLSearchParams();
        params.append("numbers", JSON.stringify(numbers));
        new objects.SetRequest('vs/timedatalog', params.toString()).perform(async);
    }

    const requestFullQualityFrame = (() => {
        const fullQualityFrameTimeout = 500
        let timeoutID
        return function requestFullQualityFrame(needFullQualityFrame) {
            if (timeoutID) {
                clearTimeout(timeoutID)
                timeoutID = undefined
            }
            if (needFullQualityFrame) {
                timeoutID = setTimeout(() => {
                    if (websocket)
                        websocket.send('F');
                        timeoutID = undefined
                    },
                    fullQualityFrameTimeout)
            }
        }
    })()

    objects.noNewFrame = false;
    let frMemNum = -1;

    function handleTextMsg(message) {
        let type = message[0];
        if (type === 'n' || type === 'F') {
            const list = message.split("##")
            let numbers = [];
            for(let i = 1; i < 6; ++i)
                numbers.push(+list[i]);
            if (websocket && type === 'n')
                websocket.send(`n:${numbers[0]}:${numbers[1]}`)
            outputMeasurements(numbers);
            drawImage(list[6]);
            if(!objects.noNewFrame && numbers[0] != frMemNum) {
                objects.noNewFrame = true;
                frMemNum = numbers[0];
            }
            requestFullQualityFrame(numbers[4] < 100 && !objects.timeService.player.isPlaying)
        }
    }

    async function handleBinaryMsg(message) {
        let ab = await message.arrayBuffer()
        let dv = new DataView(ab)
        let pos = 0
        let getUint8  = () => { const result = dv.getUint8 (pos, true); pos += 1; return result; }
        let getUint32 = () => { const result = dv.getUint32(pos, true); pos += 4; return result; }
        let getUint64 = () => {
            const lo = getUint32()
            const hi = getUint32()
            return hi * 0xffffffff + lo
        }
        let type = String.fromCharCode(getUint8())
        const n1 = getUint64()
        const n2 = getUint64()
        const level = getUint32()
        const durationMs = getUint64()
        const quality = getUint32()
        outputMeasurements([n1, n2, level, durationMs, quality]);
        let src = ab.slice(pos)
        let blob = new Blob([src], {type: 'image/jpeg'});
        let imageUrl = URL.createObjectURL(blob);
        if (websocket && type === 'n')
            websocket.send(`n:${n1}:${n2}`)
        drawImage(imageUrl)
        if(!objects.noNewFrame && n1 != frMemNum) {
            objects.noNewFrame = true;
            frMemNum = n1;
        }
        requestFullQualityFrame(quality < 100 && !objects.timeService.player.isPlaying)
    }

    function handleMsg(message) {
    if (message instanceof Blob)
        handleBinaryMsg(message)
    else
        handleTextMsg(message)
}

    try {
        if( typeof MozWebSocket == "function" )
            WebSocket = MozWebSocket;
        websocket = new WebSocket(wsUri);
        websocket.onopen = ()=> {
            websocket.send('B');    // Switch to binary frames
            websocket.send(`s:${objects.sourceId}`);
        }
        websocket.onmessage = (e)=> handleMsg(e.data);
        websocket.onerror = event => {
            objects.showError("Websocket error");
            console.error("WebSocket error observed:", event);
        }
    } catch (error) {
        objects.showError(error.message);
    }
    
})()
