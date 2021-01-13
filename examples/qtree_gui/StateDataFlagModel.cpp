#include "StateDataFlagModel.h"

namespace {

template<class T>
StateDataFlagModel::Item makeItem(const QString& name, T& prop)
{
    return {
        name,
        [&prop]() { return prop.get(); },
        [&prop](bool x) { prop.set(x); }
    };
}

template<class T>
void addItem(const QString& name, std::vector<StateDataFlagModel::Item>& items, StateData& sd) {
    items.emplace_back(makeItem(name, static_cast<T&>(sd)));
}

} // anonymous namespace

StateDataFlagModel::StateDataFlagModel(StateData& stateData, QObject *parent) :
    QAbstractListModel(parent),
    m_stateData(stateData)
{
    addItem<WithDisplayQtree>(tr("Quadtree"), m_items, m_stateData);
    addItem<WithQtreeCompactView>(tr("Compact quadtree"), m_items, m_stateData);
    addItem<WithQtreeDisplayIds>(tr("Quadtree ids"), m_items, m_stateData);
    addItem<WithLimitedQtree>(tr("Limit quadtree"), m_items, m_stateData);
    addItem<WithDisplayMesh>(tr("Original mesh"), m_items, m_stateData);
    addItem<WithDisplayBoundary>(tr("Mesh boundary"), m_items, m_stateData);
    addItem<WithDisplayBoundaryNodeNumbers>(tr("Boundary node numbers"), m_items, m_stateData);
    addItem<WithDisplayBoundaryDirMarkers>(tr("Boundary direction markers"), m_items, m_stateData);
    addItem<WithDisplaySparseNodes>(tr("Sparse nodes"), m_items, m_stateData);
    addItem<WithDisplaySparseNodeNumbers>(tr("Sparse node numbers"), m_items, m_stateData);
    addItem<WithDisplayDenseField>(tr("Dense field"), m_items, m_stateData);
    addItem<WithFill>(tr("Fill"), m_items, m_stateData);
    addItem<WithQuadtreeNodeFill>(tr("Quadtree node fill"), m_items, m_stateData);
}

int StateDataFlagModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid()? 0: static_cast<int>(m_items.size());
}

int StateDataFlagModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid()? 0: 2;
}

QVariant StateDataFlagModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
        switch (section) {
        case 0:
            return tr("Name");
        case 1:
            return tr("Value");
        default:
            Q_ASSERT(false);
            return QVariant();
        }
    }
    else
        return QVariant();
}

QVariant StateDataFlagModel::data(const QModelIndex &index, int role) const
{
    if (index.isValid()) {
        auto& item = m_items.at(index.row());
        if (index.column() == 0)
            return role == Qt::DisplayRole? item.name: QVariant();
        else {
            Q_ASSERT(index.column() == 1);
            return role == Qt::CheckStateRole? (item.get()? Qt::Checked: Qt::Unchecked): QVariant();
        }
    }
    else
        return QVariant();
}

bool StateDataFlagModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    Q_ASSERT(index.isValid() && index.column() == 1);
    switch(role) {
    case Qt::EditRole:
        m_items.at(index.row()).set(value.toBool());
        emit dataChanged(index, index);
        return true;
    case Qt::CheckStateRole:
        m_items.at(index.row()).set(value.value<Qt::CheckState>());
        emit dataChanged(index, index);
        return true;
    default:
        return false;
    }
}

Qt::ItemFlags StateDataFlagModel::flags(const QModelIndex &index) const
{
    if (index.isValid()) {
        switch(index.column()) {
        case 0:
            return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
        case 1:
            return Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsEditable| Qt::ItemIsUserCheckable;
        default:
            Q_ASSERT(false);
            return Qt::NoItemFlags;
        }
    }
    else
        return Qt::NoItemFlags;
}
