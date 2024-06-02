#include "MainWindow.h"
#include "StateDataFlagModel.h"
#include "FieldListView.h"
#include "Scene2d.h"

#include <QKeyCombination>
#include <QMenuBar>
#include <QFileDialog>
#include <QDockWidget>
#include <QTreeView>
#include <QHeaderView>
#include <QPdfWriter>
#include <QtSvg/QSvgGenerator>
#include <QMessageBox>
#include <QRegularExpression>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    auto fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(
        tr("&Open"),
        QKeyCombination(Qt::CTRL, Qt::Key_O),
        [&]()
        {
            auto fileName = QFileDialog::getOpenFileName(
                        this,
                        "Open original solution file",
                        QString(),
                        tr("Any suitable files (*.tec *.bin);;Tecplot files (*.tec);;Binary files (*.bin)"));
            if (!fileName.isEmpty())
                m_stateDataController.openInputFile(fileName);
        });

    fileMenu->addAction(
        tr("&Save view as image..."),
        QKeyCombination(Qt::CTRL, Qt::Key_S),
        [&]()
        {
            QString selectedFilter;
            auto fileName = QFileDialog::getSaveFileName(
                        this,
                        "Save view as a picture",
                        QString(),
                        tr("Svg image (*.svg);;Pdf image (*.pdf);;Png image (*.png)"),
                        &selectedFilter);
            if (!fileName.isEmpty()) {
                const char *suffixes[] = { ".pdf", ".svg", ".png" };
                bool hasSuffix = false;
                for (auto suffix : suffixes)
                    if (fileName.endsWith(suffix)) {
                        hasSuffix = true;
                        break;
                    }
                static const QRegularExpression rx("^[^*]*\\*(\\.\\w+).*$");
                auto match = rx.match(selectedFilter);
                if (!hasSuffix && match.hasMatch()) {
                    auto suffix = match.capturedTexts()[1];
                    if (!fileName.endsWith(suffix))
                        fileName += suffix;
                }
                auto cw = centralWidget();
                if (fileName.endsWith(".pdf")) {
                    QPdfWriter pix(fileName);
                    constexpr const int PdfResolution = 300;
                    pix.setResolution(PdfResolution);
                    pix.setPageSize(QPageSize(cw->size()*25.4/PdfResolution, QPageSize::Millimeter));
                    pix.setPageMargins(QMarginsF(0,0,0,0), QPageLayout::Millimeter);
                    cw->render(&pix);
                }
                else if (fileName.endsWith(".svg")) {
                    QSvgGenerator pix;
                    pix.setFileName(fileName);
                    pix.setSize(cw->size());
                    pix.setViewBox(cw->rect());
                    pix.setTitle(tr("qtree_gui view"));
                    cw->render(&pix);
                }
                else {
                    QImage pix(cw->size(), QImage::Format_RGB32);
                    cw->render(&pix);
                    if (!pix.save(fileName))
                        QMessageBox::critical(this, QString(), tr("Failed to save view to file '%1'").arg(fileName));
                }
            }
        });

    fileMenu->addSeparator();
    fileMenu->addAction(
        tr("&Quit"),
        QKeyCombination(Qt::CTRL, Qt::Key_Q),
        this,
        &QMainWindow::close);

    auto addDock = [this](Qt::DockWidgetArea dockArea, const QString& name, QWidget *widget) {
        auto dock = new QDockWidget(name, this);
        dock->setWidget(widget);
        addDockWidget(dockArea, dock);
    };

    // Add view for boolean properties
    {
        auto stateDataFlagList = new QTreeView;
        auto stateDataFlagModel = new StateDataFlagModel(m_stateDataController.stateData(), this);
        stateDataFlagList->setModel(stateDataFlagModel);
        stateDataFlagList->setIndentation(0);
        stateDataFlagList->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
        addDock(Qt::LeftDockWidgetArea, tr("Flags"), stateDataFlagList);
    }

    // Add field list view
    {
        auto fieldList = new FieldListView(m_stateDataController.stateData(), this);
        addDock(Qt::LeftDockWidgetArea, tr("Field list"), fieldList);
    }

    setCentralWidget(new Scene2d(&m_stateDataController));
}

StateData& MainWindow::stateData() {
    return m_stateDataController.stateData();
}

const StateData& MainWindow::stateData() const {
    return m_stateDataController.stateData();
}
