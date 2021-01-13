#include "ViewTransformController.h"

#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QEasingCurve>

ViewTransformController::ViewTransformController(QWidget *host) :
    QObject(host),
    m_host(host)
{
    host->installEventFilter(this);
}

void ViewTransformController::transformPainter(QPainter& painter) const
{
    QRectF rcf = m_host->rect();
    auto size = rectSize(rcf);
    auto center = rcf.center() + (size*m_currentTransform.zoom)*m_currentTransform.offset;
    painter.translate(center);
    painter.scale(m_currentTransform.zoom, m_currentTransform.zoom);
    painter.translate(-rcf.center());
}

bool ViewTransformController::eventFilter(QObject *watched, QEvent *event)
{
    Q_ASSERT(watched == m_host);
    switch (event->type()) {
    case QEvent::MouseButtonPress: {
        auto mouseEvent = static_cast<QMouseEvent*>(event);
        if (m_state == None && mouseEvent->button() == Qt::LeftButton && mouseEvent->modifiers() == Qt::NoModifier) {
            m_state = WaitDragging;
            m_dragStartRect = m_host->rect();
            m_dragStartPos = mouseEvent->pos();
        }
        break;
    }
    case QEvent::MouseButtonRelease: {
        auto mouseEvent = static_cast<QMouseEvent*>(event);
        if (m_state != None && mouseEvent->button() == Qt::LeftButton && mouseEvent->modifiers() == Qt::NoModifier)
            m_state = None;
        break;
    }
    case QEvent::MouseMove: {
        auto mouseEvent = static_cast<QMouseEvent*>(event);
        const int DragSensitivity = 3;
        if (m_state == WaitDragging && (mouseEvent->pos() - m_dragStartPos).manhattanLength() > DragSensitivity) {
            m_state = Dragging;
            m_dragStartPos = mouseEvent->pos();
            m_dragStartTransform = m_currentTransform;
            m_zooming = false;
            return true;
        }
        else if (m_state == Dragging) {
            m_endTransform.offset =
                    m_dragStartTransform.offset +
                    QPointF(mouseEvent->pos() - m_dragStartPos) / (rectSize(m_dragStartRect)*m_dragStartTransform.zoom);
            animate();
            return true;
        }
        break;
    }
    case QEvent::Wheel: {
        auto wheelEvent = static_cast<QWheelEvent*>(event);
        auto dz = exp(wheelEvent->angleDelta().y()/120. * log(2)/3);
        auto& t0 = m_timerId == 0? m_currentTransform: m_endTransform;
        m_endTransform.zoom = t0.zoom*dz;
        m_zoomStartPos = wheelEvent->pos();
        if (m_timerId == 0)
            m_dragStartRect = m_host->rect();
        m_zooming = true;
        animate();
        return true;
    }
    case QEvent::KeyPress: {
        auto keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Escape && keyEvent->modifiers() == Qt::NoModifier) {
            m_endTransform = TransformData();
            m_zooming = false;
            animate();
            return true;
        }
        break;
    }
    default:
        break;
    }
    return false;
}

void ViewTransformController::timerEvent(QTimerEvent *event)
{
    if (event->timerId() == m_timerId) {
        auto animationStepCount = AnimationDuration / AnimationTimerInterval;
        m_animationPhase += 1. / animationStepCount;
        if (m_animationPhase >= 1) {
            m_animationPhase = 1;
            killTimer(m_timerId);
            m_timerId = 0;
        }
        auto t = QEasingCurve(QEasingCurve::OutQuad).valueForProgress(m_animationPhase);
        m_currentTransform.zoom = m_startTransform.zoom*(1-t) + m_endTransform.zoom*t;

        if (m_zooming) {
            auto center = m_dragStartRect.center();
            QPointF dX = m_zoomStartPos - center;
            auto factor = (1/m_currentTransform.zoom - 1/m_startTransform.zoom) / rectSize(m_dragStartRect);
            m_currentTransform.offset = m_startTransform.offset + dX*factor;
        }
        else
            m_currentTransform.offset = m_startTransform.offset*(1-t) + m_endTransform.offset*t;

        m_host->update();
    }
}

void ViewTransformController::animate()
{
    if (m_timerId == 0)
        m_timerId = startTimer(AnimationTimerInterval);
    m_startTransform = m_currentTransform;
    m_animationPhase = 0;
}
