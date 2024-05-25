#pragma once

#include "StateData.h"

#include <QWidget>

class ViewTransformController :
        public QObject
{
Q_OBJECT
public:
    explicit ViewTransformController(QWidget *host);

    void transformPainter(QPainter& painter) const;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;
    void timerEvent(QTimerEvent *event) override;

private:
    QWidget *m_host;

    struct TransformData
    {
        double zoom = 1;
        QPointF offset = { 0, 0 };
    };
    TransformData m_currentTransform;

    enum State {
        None,
        WaitDragging,
        Dragging
    };
    State m_state = None;
    QRect m_dragStartRect;
    QPointF m_dragStartPos;
    QPointF m_zoomStartPos = { 0, 0 };
    TransformData m_dragStartTransform;
    TransformData m_startTransform;
    TransformData m_endTransform;
    bool m_zooming = false;

    template<class R>
    static double rectSize(
            const R& rect,
            std::enable_if_t<std::is_same<R, QRect>::value || std::is_same<R, QRectF>::value, int> = 0)
    {
        return std::min(rect.width(), rect.height());
    }

    double m_animationPhase = 1;
    int m_timerId = 0;
    const int AnimationTimerInterval = 40;
    const int AnimationDuration = 400;
    void animate();
};
