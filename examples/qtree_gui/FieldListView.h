#pragma once

#include <QListView>

#include "StateData.h"

class FieldListView : public QListView
{
Q_OBJECT
public:
    explicit FieldListView(StateData& stateData, QWidget *parent = nullptr);

private:
    StateData& m_stateData;
    void selectCurrentField();
};
