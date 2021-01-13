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
const cookie = require('cookie');
const _ = require('lodash')
const fs = require('fs');
const vs_input = require('./vs_input')
const vs_time = require('./vs_time')
const vs_fields = require('./vs_fields')
const vs_camera = require('./vs_camera')
const vs_mouse = require('./vs_mouse')
const vs_problems = require('./vs_problems');
const vs_logout = require('./vs_logout');
const vs_login = require('./vs_login');
const vs_alive = require('./vs_alive');
const vs_timedatalog = require('./vs_timedatalog');
const vs_share = require('./vs_share');
const s3vs = require('../../s3vs_js/build/Release/s3vs_js.node')
const random_string = require('../modules/random_string');
const frame_server = require('../modules/frame_server');

const s3vs_binary_dir = process.env.s3vs_binary_dir
if (!s3vs_binary_dir)
    throw new Error('Please specify the s3vs_binary_dir environment variable')
const kill_vsc_interval = process.env.kill_vsc_interval || 60000;

class UserData 
{
    constructor() {
        this.vsc = null;
        this.initializeVsController();
        this.sourceId = random_string()
        this.updateFrameSource()
    }
    initializeVsController() {
        this.vsc = new s3vs.VsController(s3vs_binary_dir);
        this.loadVsControllerConfig();
        this.vsc.start();
        this.vsc.isKilled = false;

        this.updateLastTimeAlive();
        let intervalCb = ()=>{
            let now = Date.now();
            let f = 1.5;   // to avoid kind of 'kill races'
            if(now - this.vsc.lastTimeAlive > kill_vsc_interval * f)
                this.killVsController();
        }
        this.vsc.killTimerId = setInterval(
            intervalCb.bind(this), kill_vsc_interval );
    }
    loadVsControllerConfig() {
        const configFileName = process.env.s3vs_config_file || './configs/vs_controller_config.json'
        const cfg = JSON.parse(fs.readFileSync(configFileName, 'utf8'))
        const cp = (dst, src) => {
            _.each(src, (v, k) => {
                if (v instanceof Object)
                    cp(dst[k], v)
                else
                    dst[k] = v
            })
        }
        cp(this.vsc, cfg)
    }
    updateFrameSource() {
        const frameOutput = this.vsc.frameOutput
        frame_server.setSource(this.sourceId, frameOutput.shmem, [frameOutput.frameWidth, frameOutput.frameHeight])
    }
    killVsController() {
        console.log("KillVsController");
        clearInterval(this.vsc.killTimerId);
        this.vsc.kill();
        this.vsc.isKilled = true;
    }
    updateLastTimeAlive() {
        let now = Date.now();
        this.vsc.lastTimeAlive = now;
    }
}

let userData = {} // Key = userId, value = UserData

// Release shared memory on exit
process.on('SIGINT', function() {
    console.log("Caught interrupt signal");
    _.each(userData, d => d.vsc.kill())
    process.exit();
})

const router = express.Router()

router
    .use((req, res, next) => {
        let d = userData[req.cookies.userId]
        if (!d) {
            d = userData[req.cookies.userId] = new UserData
            console.log(`Created s3vs server, userId: ${req.cookies.userId}, sourceId: ${d.sourceId}`)
        } else {
            if(d.vsc.isKilled) {
                console.log("UserData is specified but VsController is killed");
                d.initializeVsController();
            }
        }
        res.setHeader('Set-Cookie', cookie.serialize('sourceId', d.sourceId, {
            httpOnly: true,
            maxAge: 60 * 60 * 24 * 365 // 1 year
        }));
        req.userData = d
        next()
    })
    .use('/sourceId', (req, res)=> res.send(JSON.stringify(req.userData.sourceId)) )
    .use('/input', vs_input)
    .use('/time', vs_time)
    .use('/fields', vs_fields)
    .use('/camera', vs_camera)
    .use('/mouse', vs_mouse)
    .use('/problems', vs_problems)
    .use('/logout', vs_logout)
    .use('/login', vs_login)
    .use('/alive', vs_alive)
    .use('/timedatalog', vs_timedatalog)
    .use('/share', vs_share)

module.exports = router
