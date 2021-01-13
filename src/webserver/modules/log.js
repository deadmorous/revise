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

const fs = require("fs")

const requests_log_path = process.env.requests_log_path;
if (!requests_log_path)
    throw new Error('Please specify the requests_log_path environment variable');
const timedata_log_path = process.env.timedata_log_path;
if (!timedata_log_path)
    throw new Error('Please specify the timedata_log_path environment variable');

let log_fn = {};

log_fn.request = function(req) {
    let time = Date.now();
    let userId = req.cookies.userId;
    let requestStr = req.originalUrl;
    let dataStr = '';
    dataStr += `timeSinceEpoch: ${time}; `
    dataStr += `userId: ${userId}; `;
    dataStr += `requestStr: ${requestStr}\n`;
    fs.appendFileSync(requests_log_path, dataStr);
}

log_fn.timeData = function(timeData) {
    let dataStr = "";
    dataStr += `Shared Memory Frame Number: ${timeData[0]}\n`;
    dataStr += `Frame Level: ${timeData[1]}\n`;
    dataStr += `Rendering Duration: ${timeData[2]}\n`;
    dataStr += `Client Frame Number: ${timeData[3]}\n`;
    dataStr += `\n`;
    fs.appendFileSync(timedata_log_path, dataStr);
}

module.exports = log_fn;