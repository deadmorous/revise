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
const s3vs = require('../../s3vs_js/build/Release/s3vs_js.node')

const s3vs_binary_dir = process.env.s3vs_binary_dir
if (!s3vs_binary_dir)
    throw new Error('Please specify the s3vs_binary_dir environment variable')

router.get("/", (req, res)=>{
    try {
        if(!req.userData.vsc.isKilled)
            throw new Error("VsController is not killed");
        req.userDate.initializeVsController();
    } catch(error) {
        console.log(error.message);
        res.status(400).send(error.message);
    }
})

module.exports = router;