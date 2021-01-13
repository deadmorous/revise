#include "StateDataController.h"

StateDataController::StateDataController(QObject *parent) :
    QObject(parent)
{
    m_stateData.onInputFileNameChanged([this](const std::string& name) {
        emit inputFileChanged(QString::fromStdString(name));
    });

    auto emitQtreeViewChanged = [this](){ emit qtreeViewChanged(); };
    m_stateData.onInputFileNameChanged(emitQtreeViewChanged);
    m_stateData.onFieldNameChanged(emitQtreeViewChanged);
    m_stateData.onBlockTreeLocationChanged(emitQtreeViewChanged);
    m_stateData.onDisplayQtreeChanged(emitQtreeViewChanged);
    m_stateData.onQtreeCompactViewChanged(emitQtreeViewChanged);
    m_stateData.onQtreeDisplayIdsChanged(emitQtreeViewChanged);
    m_stateData.onLimitedQtreeChanged(emitQtreeViewChanged);
    m_stateData.onDisplayMeshChanged(emitQtreeViewChanged);
    m_stateData.onDisplayBoundaryChanged(emitQtreeViewChanged);
    m_stateData.onDisplayBoundaryNodeNumbersChanged(emitQtreeViewChanged);
    m_stateData.onDisplayBoundaryDirMarkersChanged(emitQtreeViewChanged);
    m_stateData.onDisplaySparseNodesChanged(emitQtreeViewChanged);
    m_stateData.onDisplaySparseNodeNumbersChanged(emitQtreeViewChanged);
    m_stateData.onDisplayDenseFieldChanged(emitQtreeViewChanged);
    m_stateData.onFillChanged(emitQtreeViewChanged);
    m_stateData.onQuadtreeNodeFillChanged(emitQtreeViewChanged);
}

StateData& StateDataController::stateData() {
    return m_stateData;
}

const StateData& StateDataController::stateData() const {
    return m_stateData;
}

void StateDataController::openInputFile(const QString& fileName) {
    m_stateData.setInputFileName(fileName.toStdString());
}
