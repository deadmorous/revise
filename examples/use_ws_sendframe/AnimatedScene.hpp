#pragma once

#include <QImage>

#include <memory>

class AnimatedScene
{
public:
    AnimatedScene(int width, int height);
    ~AnimatedScene();

    void advance();
    QImage render();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
