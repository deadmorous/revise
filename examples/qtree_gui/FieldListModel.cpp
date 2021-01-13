#include "FieldListModel.h"

#include <boost/range/algorithm/transform.hpp>

FieldListModel::FieldListModel(StateData& stateData, QObject *parent) :
    QAbstractListModel(parent),
    m_stateData(stateData)
{
    loadModel();

    m_stateData.onInputFileNameChanged([this]() {
        beginResetModel();
        loadModel();
        endResetModel();
    });
}

void FieldListModel::loadModel()
{
    if (!m_stateData.inputFileName().empty()) {
        auto& fs = m_stateData.fieldService();
        m_fields.resize(fs.fieldCount());
        boost::range::transform(fs.fieldNames(), m_fields.begin(), [](const std::string& s) {
            return QString::fromStdString(s);
        });
    }
    else
        m_fields.clear();
}

int FieldListModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid()? 0: static_cast<int>(1 + m_fields.size());
}

QVariant FieldListModel::data(const QModelIndex &index, int role) const
{
    if (index.isValid() && role == Qt::DisplayRole) {
        if (index.row() == 0)
            return tr("<no field>");
        else
            return m_fields.at(index.row()-1);
    }
    else
        return QVariant();
}

Qt::ItemFlags FieldListModel::flags(const QModelIndex &index) const
{
    if (index.isValid()) {
        switch(index.column()) {
        case 0:
            return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
        default:
            Q_ASSERT(false);
            return Qt::NoItemFlags;
        }
    }
    else
        return Qt::NoItemFlags;
}
