#pragma once

#include <QWidget>
#include <boost/any.hpp>

class StateDataController;
class ViewTransformController;

class Scene2d : public QWidget
{
Q_OBJECT
public:
    explicit Scene2d(StateDataController *stateDataController, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    StateDataController *m_stateDataController;
    ViewTransformController *m_transformController;

    boost::any m_sparseNodesCache;
};
