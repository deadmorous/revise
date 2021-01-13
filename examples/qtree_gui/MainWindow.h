#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "StateDataController.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

    StateData& stateData();
    const StateData& stateData() const;

signals:

private:
    StateDataController m_stateDataController;
};

#endif // MAINWINDOW_H
