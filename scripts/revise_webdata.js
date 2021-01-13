#!/usr/bin/env node

const path = require('path')
const fs = require('fs')

const scriptDir = path.dirname(process.argv[1])
const args = process.argv.slice(2)

const webserverDir = path.normalize(path.join(scriptDir, '..', 'src', 'webserver'))

const vsEnvFileName = path.join(scriptDir, 'custom', 'webserver_env.sh')
const rxProblemListConfigLine = /^export s3vs_problem_list_file=\$S3DMM_CUSTOM_SCRIPTS_DIR\/(.*)$/

function installedProblemListFileName() {
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
        const m = line.match(rxProblemListConfigLine)
        if (m)
            result = m[1]
    })
    if (!result)
        throw new Error('No problem list file is installed, use the install command')
    return result
}

function installProblemListFile(problemListFileName) {
    let vsEnvFileContents
    try {
        vsEnvFileContents = fs.readFileSync(vsEnvFileName, 'utf8')
        let lines = vsEnvFileContents.split(/\r?\n/)
        let matched = false
        const problemListConfigLine = `export s3vs_problem_list_file=$S3DMM_CUSTOM_SCRIPTS_DIR/${problemListFileName}`
        for (let i=0, n=lines.length; i<n; ++i) {
            let line = lines[i]
            const m = line.match(rxProblemListConfigLine)
            if (m) {
                if (matched) {
                    lines.splice(i, 1)
                    --i
                    --n
                }
                else {
                    lines[i] = problemListConfigLine
                    matched = true
                }
            }
        }
        if (!matched)
            lines.push(problemListConfigLine)
        vsEnvFileContents = lines.join('\n')
    }
    catch(e) {
        vsEnvFileContents = `#!/bin/bash

export s3vs_problem_list_file=$S3DMM_CUSTOM_SCRIPTS_DIR/${problemListFileName}
`
    }
    fs.writeFileSync(vsEnvFileName, vsEnvFileContents)
    
    let problemListFilePath = path.join(scriptDir, 'custom', problemListFileName)
    try {
        fs.accessSync(problemListFilePath, fs.constants.R_OK);
    }
    catch(e) {
        fs.writeFileSync(problemListFilePath, '{ "problems": [] }')
    }
}

function uninstallProblemListFile() {
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
        const m = line.match(rxProblemListConfigLine)
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

function readProblemList() {
    return JSON.parse(fs.readFileSync(path.join(scriptDir, 'custom', installedProblemListFileName()), 'utf8'))
}

function saveProblemList(o) {
    fs.writeFileSync(
        path.join(scriptDir, 'custom', installedProblemListFileName()),
        JSON.stringify(o, null, 4))
}

function absPath(baseDir, pathName) {
    return path.isAbsolute(pathName)? pathName: path.normalize(path.join(baseDir, pathName))
}

function addProblem(pathName, title) {
    let o = readProblemList();
    let absPathName = absPath(process.cwd(), pathName)
    o.problems.forEach(p => {
        if (absPath(webserverDir, p.path) === absPathName)
            throw new Error(`Problem with path '${pathName}' has already been specified and is titled '${p.name}'`)
        if (p.name === title)
            throw new Error(`Problem with title '${title}' has already been specified and has path '${p.path}'`)
    })
    let ultimatePath = path.isAbsolute(pathName)? pathName: path.relative(webserverDir, absPathName)
    o.problems.push({name: title, path: ultimatePath})
    saveProblemList(o)
}

function removeProblem(criterion) {
    let o = readProblemList()
    let oldProblemCount = o.problems.length
    o.problems = o.problems.filter(criterion)
    if (o.problems.length != oldProblemCount)
        saveProblemList(o)
    else
        throw new Error('The problem specified could not be found')
}

function removeProblemByPath(pathName) {
    let absPathName = absPath(process.cwd(), pathName)
    return removeProblem(p => absPath(webserverDir, p.path) !== absPathName)
}

function removeProblemByTitle(title) {
    return removeProblem(p => p.name !== title)
}

function clearProblems() {
    let o = readProblemList()
    if (o.problems.length > 0) {
        o.problems = []
        saveProblemList(o)
    }
}

function listProblems() {
    readProblemList().problems.forEach(p => {
        process.stdout.write(`${p.name}\n${p.path}\n\n`)
    })
}

function displayHelp() {
    process.stdout.write([
        'This script helps managing dataset accessible via the ReVisE web server.',
        'Usage: revise_webdata command [args]',
        '(args depend on command; some commands do not require arguments).',
        'The following commands are available.',
        '',
        'install <list_name>',
        '  Installs dataset list file for use by the web server. The file has the JSON format,',
        '  so it is a good idea to give it the .json filename extension.',
        '  Commands that modify or read the dataset list always operate on the currently',
        '  installed list file.',
        '  Notice that <list_name> should not contain any path.',
        '',
        'uninstall',
        '  Uninstalls dataset list file for use by the web server.',
        '  After this command, the webserver uses the default dataset list file.',
        '',
        'installed',
        '  Displays the name of currently installed dataset list file.',
        '',
        'add <path> <title>',
        '  Adds dataset with specified path to the main dataset file and specified title to the list',
        '  <path> may be either absolute or relative to the current working directory.',
        '',
        'remove <path>',
        '  Removes dataset with the specified path from the list',
        '',
        'remove_by_title <title>',
        '  Removes dataset with the specified title from the list',
        '',
        'clear',
        '  Removes all datasets from the list',
        '',
        'list',
        '  Displays paths and titles of all datasets in the list',
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
        installProblemListFile(args[0])
    },
    uninstall: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        uninstallProblemListFile()
    },
    installed: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        console.log(installedProblemListFileName())
    },
    add: args => {
        if (args.length != 2)
            throw new Error(`2 arguments were expected (path and title), ${args.length} provided`)
        addProblem(args[0], args[1])
    },
    remove: args => {
        if (args.length != 1)
            throw new Error(`1 argument was expected (path), ${args.length} provided`)
        removeProblemByPath(args[0])
    },
    remove_by_title: args => {
        if (args.length != 1)
            throw new Error(`1 argument was expected (path), ${args.length} provided`)
        removeProblemByTitle(args[0])
    },
    clear: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        clearProblems()
    },
    list: args => {
        if (args.length != 0)
            throw new Error(`No arguments were expected, ${args.length} provided`)
        listProblems()
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
