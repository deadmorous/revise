#pragma once

#include <QAbstractListModel>

#include "StateData.h"

class StateDataFlagModel : public QAbstractListModel
{
Q_OBJECT
public:
    explicit StateDataFlagModel(StateData& stateData, QObject *parent = nullptr);

    int rowCount(const QModelIndex& parent) const override;
    int columnCount(const QModelIndex& parent) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    struct Item
    {
        QString name;
        std::function<bool()> get;
        std::function<void(bool)> set;
    };

private:
    StateData& m_stateData;

    std::vector<Item> m_items;
};
