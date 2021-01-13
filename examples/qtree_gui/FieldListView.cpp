#include "FieldListView.h"
#include "FieldListModel.h"

#include <boost/range/algorithm/find.hpp>

FieldListView::FieldListView(StateData& stateData, QWidget *parent) :
    QListView(parent),
    m_stateData(stateData)
{
    auto model = new FieldListModel(m_stateData, this);
    setModel(model);

    auto selection = selectionModel();

    selectCurrentField();

    connect(selection, &QItemSelectionModel::currentRowChanged, [&](const QModelIndex &current, const QModelIndex &previous) {
        auto fieldName = current.row() == 0? std::string(): current.data().toString().toStdString();
        if (fieldName != m_stateData.fieldName())
            m_stateData.setFieldName(fieldName);
    });

    connect(model, &FieldListModel::modelReset, this, &FieldListView::selectCurrentField);
}

void FieldListView::selectCurrentField()
{
    int currentIndex = 0;
    if (m_stateData.hasSparseField()) {
        auto idx = m_stateData.fieldService().maybeFieldIndex(m_stateData.fieldName());
        if (idx != ~0u)
            currentIndex = 1 + static_cast<int>(idx);
    }
    setCurrentIndex(model()->index(currentIndex, 0));
}
