#!/usr/bin/env node

const path = require('path')
const fs = require('fs')

const scriptDir = path.dirname(process.argv[1])
const args = process.argv.slice(2)

const vsEnvFileName = path.join(scriptDir, 'custom', 'webserver_env.sh')
const rxHwConfigLine = /^export s3vs_config_file=\$S3DMM_CUSTOM_SCRIPTS_DIR\/(.*)$/

function installedHwConfigFileName() {
    let vsEnvFileContents
    try {
        vsEnvFileContents = fs.readFileSync(vsEnvFileName, 'utf8')
    }
    catch(e) {
        console.error(`File '${vsEnvFileName}' could not be opened, try using the install command.`)
        throw e
    }
    let result
    vsEnvFileContents.split(/\r?\n/).forEach(line => {
        const m = line.match(rxHwConfigLine)
        if (m)
            result = m[1]
    })
    if (!result)
        throw new Error('No problem list file is installed, use the install command')
    return result
}

function installHwConfigFile(hwConfigFileName) {
    let vsEnvFileContents
    try {
        vsEnvFileContents = fs.readFileSync(vsEnvFileName, 'utf8')
        let lines = vsEnvFileContents.split(/\r?\n/)
        let matched = false
        const hwConfigLine = `export s3vs_config_file=$S3DMM_CUSTOM_SCRIPTS_DIR/${hwConfigFileName}`
        for (let i=0, n=lines.length; i<n; ++i) {
            let line = lines[i]
            const m = line.match(rxHwConfigLine)
            if (m) {
                if (matched) {
                    lines.splice(i, 1)
                    --i
                    --n
                }
                else {
                    lines[i] = hwConfigLine
                    matched = true
                }
            }
        }
        if (!matched)
            lines.push(hwConfigLine)
        vsEnvFileContents = lines.join('\n')
    }
    catch(e) {
        vsEnvFileContents = `#!/bin/bash

export s3vs_config_file=$S3DMM_CUSTOM_SCRIPTS_DIR/${hwConfigFileName}
`
    }
    fs.writeFileSync(vsEnvFileName, vsEnvFileContents)
    
    let hwConfigFilePath = path.join(scriptDir, 'custom', hwConfigFileName)
    try {
        fs.accessSync(hwConfigFilePath, fs.constants.R_OK);
    }
    catch(e) {
        fs.writeFileSync(hwConfigFilePath, JSON.stringify(
            {
                computingCaps: {
                    compNodeCount: 1,
                    GPUPerNodeCount: 1,
                    CPUPerNodeCount: 1,
                    workerThreadPerNodeCount: 1
                },
                controllerOpts: {
                    measureRenderingTime: true
                }
            },
            null, 4
        ))
    }
}

function uninstallHwConfigFile() {
    let vsEnvFileContents
    try {
        vsEnvFileContents = fs.readFileSync(vsEnvFileName, 'utf8')
    }
    catch(e) {
        return
    }
    let lines = vsEnvFileContents.split(/\r?\n/)
    let matched = false
    for (let i=0, n=lines.length; i<n; ++i) {
        let line = lines[i]
        const m = line.match(rxHwConfigLine)
        if (m) {
            lines.splice(i, 1)
            --i
            --n
            matched = true
        }
    }
    if (matched)
        fs.writeFileSync(vsEnvFileName, lines.join('\n'))
}

function readHwConfig() {
    return JSON.parse(fs.readFileSync(path.join(scriptDir, 'custom', installedHwConfigFileName()), 'utf8'))
}

function saveHwConfig(o) {
    fs.writeFileSync(
        path.join(scriptDir, 'custom', installedHwConfigFileName()),
        JSON.stringify(o, null, 4))
}

const parameters = {
    nodes: {
        get: o => o.computingCaps.compNodeCount,
        set: (o, v) => o.computingCaps.compNodeCount = +v
    },
    gpus_per_node: {
        get: o => o.computingCaps.GPUPerNodeCount,
        set: (o, v) => o.computingCaps.GPUPerNodeCount = +v
    },
    render_threads_per_node: {
        get: o => o.computingCaps.workerThreadPerNodeCount,
        set: (o, v) => o.computingCaps.workerThreadPerNodeCount = +v
    },
    assemble_threads_per_node: {
        get: o => o.computingCaps.CPUPerNodeCount,
        set: (o, v) => o.computingCaps.CPUPerNodeCount = +v
    },
    measure_time: {
        get: o => o.controllerOpts.measureRenderingTime,
        set: (o, v) => o.controllerOpts.measureRenderingTime = v === 'true' || v === true
    }
}

function getParameter(name) {
    let o = readHwConfig();
    if (name in parameters)
        return parameters[name].get(o)
    else
        throw new Error(`Unknown parameter '${name}'`)
}

function setParameter(name, value) {
    let o = readHwConfig();
    if (name in parameters) {
        parameters[name].set(o, value)
        saveHwConfig(o)
    }
    else
        throw new Error(`Unknown parameter '${name}'`)
}

function displayHelp() {
    process.stdout.write([
        'This script helps configuring hardware resources for ReVisE.',
        'Usage: revise_hw_config command [args]',
        '(args depend on command; some commands do not require arguments).',
        'The following commands are available.',
        '',
        'install <hw_config_name>',
        '  Installs hw config file for use by ReVisE. The file has the JSON format,',
        '  so it is a good idea to give it the .json filename extension.',
        '  Commands that modify or read the hw config always operate on the currently',
        '  installed file.',
        '  Notice that <hw_config_name> should not contain any path.',
        '',
        'uninstall',
        '  Uninstalls hw config file for use by ReVisE.',
        '  After this command, the webserver uses the default configuration file.',
        '',
        'installed',
        '  Displays the name of currently installed hw config file.',
        '',
        'get <parameter>',
        '  Displays the current value of specified parameter; <parameter> can be one of',
        '  nodes, gpus_per_node, render_threads_per_node, assemble_threads_per_node,',
        '  measure_time.',
        '',
        'set <parameter> <value>',
        '  Sets the specified parameter to the specified value; <parameter> can be one of',
        '  nodes, gpus_per_node, render_threads_per_node, assemble_threads_per_node,',
        '  measure_time.',
        '',
        'help',
        '  Displays this help message.',
        ''
    ].join('\n'))
}

const commands = {
    install: args => {
        if (args.length != 1)
            throw new Error(`1 argument was expected (file name), ${args.length} provided`)
        installHwConfigFile(args[0])
    },
    uninstall: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        uninstallHwConfigFile()
    },
    installed: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        console.log(installedHwConfigFileName())
    },
    get: args => {
        if (args.length !=1)
            throw new Error(`1 argument was expected (name), ${args.length} provided`)
        console.log(getParameter(args[0]))
    },
    set: args => {
        if (args.length !=2)
            throw new Error(`2 arguments were expected (name and value), ${args.length} provided`)
        setParameter(args[0], args[1])
    },
    help: displayHelp,
    '--help': displayHelp
}

try {
    if (args.length < 1)
        throw new Error('A command was expected (try the help command)')
    const command = args[0]
    if (command in commands)
        commands[command](args.slice(1))
    else
        throw new Error(`Unknown command '${command}'`)
}
catch(e) {
    console.error(e.message)
    process.exit(1)
}
