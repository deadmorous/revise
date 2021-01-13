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

const express = require('express');
const router = express.Router();

function exceptionTo400(res, func) {
    try {
        func();
    }
    catch(e) {
        res.status(400).send(e.message);
        console.log(e.message);
        return;
    }
}

router
    .get('/get', (req, res)=>{
        exceptionTo400(res, ()=>{
            let cameraTransform = req.userData.vsc.cameraTransform;
            res.send( JSON.stringify(cameraTransform) );
        });
    })
    .get('/set', (req, res)=>{
        exceptionTo400(res, ()=>{
            let cameraTransform = JSON.parse( req.query.m );
            let centerPositionStr = req.query.cp;
            req.userData.vsc.cameraTransform = cameraTransform;
            if(centerPositionStr)
                req.userData.vsc.cameraCenterPosition = JSON.parse(centerPositionStr);
            res.sendStatus(200);
        });
    })

module.exports = router;