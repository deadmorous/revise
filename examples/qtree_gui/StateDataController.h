#pragma once

#include "StateData.h"

#include <QObject>

class StateDataController :
        public QObject
{
Q_OBJECT
public:
    explicit StateDataController(QObject *parent = nullptr);

    StateData& stateData();
    const StateData& stateData() const;

public slots:
    void openInputFile(const QString& fileName);

signals:
    void inputFileChanged(const QString& fileName);
    void qtreeViewChanged();

private:
    StateData m_stateData;
};
