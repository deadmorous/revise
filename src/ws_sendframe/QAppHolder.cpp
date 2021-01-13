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

#include "ws_sendframe/QAppHolder.hpp"

#include <QCoreApplication>

namespace {

int argc = 1;
char arg1[] = "ws_sendfile";
char *argv[2] = {arg1, nullptr};
QCoreApplication *app = nullptr;
unsigned int count = 0;

} // anonymous namespace

QAppHolder::QAppHolder()
{
    if (!app) {
        Q_ASSERT(count == 0);
        if (qApp) {
            app = qApp;
            ++count; // Do nothing in the case of already existing qApp
        }
        else
            app = new QCoreApplication(argc, argv);
    }
    ++count;
}

QAppHolder::~QAppHolder()
{
    if (!--count) {
        Q_ASSERT(app);
        delete app;
        app = nullptr;
    }
}
