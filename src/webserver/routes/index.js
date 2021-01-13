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
const vs = require('./vs')
const random_string = require('../modules/random_string');
const log = require('../modules/log');

const router = express.Router();

router
  .use((req,res,next)=>{
    if (!req.cookies.userId) {
      // Set a new cookie with the userId
      const userId = random_string()
      res.setHeader('Set-Cookie', cookie.serialize('userId', userId, {
        httpOnly: true,
        maxAge: 60 * 60 * 24 * 365 // 1 year
      }));
      // Redirect back after setting cookie
      res.setHeader('Location', req.headers.referer || '/');
      return res.sendStatus(302);
    }
    next();
  })
  .use((req, res, next)=>{
    log.request(req);
    next();
  })
  .use('/vs', vs)

/* GET home page. */
router
.get('/', function(req, res) {
  res.render( 'index' );
})


router.get('/userId',(req,res)=>{
  res.send("userId: " + req.cookies.userId);
})


module.exports = router;