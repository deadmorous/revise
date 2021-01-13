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

const express = require('express')
const assert = require('assert')
const _ = require('lodash')

const router = express.Router()

function bypath(d, path) {
    _.each(path, item => {
        if (d instanceof Object)
            d = d[item]
        else
            d = undefined
    })
    return d
}

function bypathString(d, pathString) {
    return bypath(d, _.toPath(pathString || ''))
}

function copyScalarValue(node, name, value)
{
    if (!(node instanceof Object && name in node))
        throw new Error('Invalid argument (property name is not found)')
    let td = typeof node[name]
    let ts = typeof value
    if (td !== ts)
        throw new Error(`Invalid argument (expected type ${td}, got ${ts} for property ${name})`)
    node[name] = value
}

function copyObjectValue(node, value)
{
    assert.ok(value instanceof Object, 'Trying to copy a scalar value to an object property')
    if (!(node instanceof Object))
        throw new Error('Invalid argument (trying to copy an object value to a scalar property)')
    _.each(value, (k, v) => {
        if (!(k in node))
            throw new Error(`Invalid argument (trying to set value of non-existing property ${k})`)
        if (v instanceof Object)
            copyObjectValue(node[k], v)
        else
            copyScalarValue(node, k, v)
    })
}

function copySimpleArrayValue(node, name, value)
{
    assert.ok(value instanceof Array, 'Trying to copy a non-array value to an Array property')
    if (!(node[name] instanceof Array))
        throw new Error('Invalid argument (trying to copy an array value to a non-array property)')
    node[name] = value    // That seems to be enough for our use case
}

function setValue (userData, input, pathString, value)
{
    let path = _.toPath(pathString || '')
    if (value instanceof Array) {
        if (path.length < 1)
            throw new Error('Invalid argument (cannot copy array value of root)')
        let name = path.splice(-1)[0]
        let node = bypath(input, path)
        copySimpleArrayValue(node, name, value)
        if (name == 'viewportSize')
            userData.updateFrameSource()
    }
    else if (value instanceof Object) {
        if (path.length >= 1) {
            let name = path.splice(-1)[0]
            let node = bypath(input, path)
            if (node[name].constructor === Object)
                node[name] = value
            else
                copyObjectValue(node[name], value)
        }
        else
            copyObjectValue(input, value)
    }
    else {
        if (path.length < 1)
            throw new Error('Invalid argument (cannot copy scalar value of root)')
        let name = path.splice(-1)[0]
        let node = bypath(input, path)
        copyScalarValue(node, name, value)
    }
}

function exceptionTo400(res, f) {
    try {
        f()
    }
    catch(e) {
        console.log(e.message)
        res.status(400).send(e.message)
        return
    }
}

router
    .get('/names', (req, res) => {
        exceptionTo400(res, () => {
            let input = req.userData.vsc.input
            let node = bypathString(input, req.query.root)
            if (!(node instanceof Object))
                throw new Error('Invalid argument (trying to get property names of a non-object)')
            res.send(JSON.stringify(_.keys(node)))
        })
    })

    .get('/get', (req, res) => {
        exceptionTo400(res, () => {
            let input = req.userData.vsc.input
            let node = bypathString(input, req.query.root)
            if (node === undefined)
                throw new Error('Invalid argument (no property at the path specified)')
            res.send(JSON.stringify(node))
        })
    })

    .get('/set', (req, res) => {
        exceptionTo400(res, () => {
            let input = req.userData.vsc.input
            let value = JSON.parse(req.query.value)
            setValue(req.userData, input, req.query.root, value)
            res.sendStatus(200)
        })
    })

module.exports = router;
