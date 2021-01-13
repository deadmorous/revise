#pragma once

#include <QAbstractListModel>

#include "StateData.h"

class FieldListModel : public QAbstractListModel
{
Q_OBJECT
public:
    explicit FieldListModel(StateData& stateData, QObject *parent = nullptr);

    int rowCount(const QModelIndex& parent) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

private:
    StateData& m_stateData;
    std::vector<QString> m_fields;
    void loadModel();
};
