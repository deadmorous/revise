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

const path = require('path')
const child_process = require('child_process')

const s3vs_binary_dir = process.env.s3vs_binary_dir
if (!s3vs_binary_dir)
    throw new Error('Please specify the s3vs_binary_dir environment variable')

const s3vs = require('./build/Release/s3vs_js.node')

console.log('exports: ', s3vs)
console.log()


let vsc = new s3vs.VsController(s3vs_binary_dir)
vsc.computingCaps.GPUPerNodeCount = 1
vsc.start()
console.log('vsc: ', vsc)
console.log('input: ', vsc.input)

// vsc.input.problemPath = '../../data/synthetic-tests/quality-tests/data/mwave_d8.coleso'
vsc.input.problemPath = '../../data/synthetic-tests/sphere/data/sphere_N-1e3_t-1_D-6_d-3/sphere_N-1e3_t-1_D-6_d-3.bin'

console.log(`timeValues: ${vsc.timeValues()}`);

vsc.cameraTransform = [
            1, 0, 0, 0,
            0, 0.961524, 0.274721, 0,
            0, -0.274721, 0.961524, 0,
           -0, -0, -36.4006, 1]
console.log(vsc.cameraTransform)

vsc.input.fieldMode = 'Isosurface'
vsc.input.clippingPlanes = [
            {pos: [1,2,3], normal: [0.1, 0, 1]}
        ]
vsc.input.fieldParam.threshold = 0.22
vsc.input.fieldParam.colorTransferFunction = {
    0:   [0,0,0,1],
    0.5: [1,0,0,0],
    1:   [1,1,1,1]
}


const frameOutput = vsc.frameOutput
console.log('frameOutput: ', frameOutput)

const wsock = child_process.spawn(
                path.join(s3vs_binary_dir, 'websock'),
                ['-s',  frameOutput.shmem,
                 '-c', frameOutput.frameWidth,
                 '-r', frameOutput.frameHeight,
                 '-f', frameOutput.frameFormat])

process.on('SIGINT', function() {
    console.log("Caught interrupt signal");
    vsc.kill()
    process.exit();
})

console.log('--------------')
process.stdout.write(JSON.stringify(vsc.input, null, 2) + '\n')
console.log('-------------- Fields')
vsc.fieldNames.forEach(fieldName => {
    let fieldRange = vsc.fieldRange(fieldName)
    console.log(`${fieldName}: ${fieldRange}`)
})
console.log('-------------- ComputingCaps')
process.stdout.write(JSON.stringify(vsc.computingCaps, null, 2) + '\n')
console.log('-------------- ControllerOpts')
process.stdout.write(JSON.stringify(vsc.controllerOpts, null, 2) + '\n')

// setTimeout(() => vsc.kill(), 10000)

//function resolveAfter(ms) {
//    return new Promise(resolve => {
//        setTimeout(() => resolve(0), ms)
//    })
//}

//async function pause(ms) {
//    const result = await resolveAfter(ms)
//    console.log(`${ms} milliseconds passed`)
//    return result
//}

//try {
//    pause(1500).then(() => {
//        console.log('input: ', vsc.input)
//        console.log('fieldNames: ', vsc.fieldNames)
//        vsc.updateMouseState({x: 1, y: 2, wheelDelta: 3, flags: 123})
//        vsc.options = {patience: 123}
//        vsc.kill()
//    })
//}
//catch(e) {
//    console.log('XXXXXXXXXXXXXXXXXXXXXXXX')
//}
