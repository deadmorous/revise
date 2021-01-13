#include "AnimatedScene.hpp"

#include <QPainter>

#include <boost/assert.hpp>

#include "Vec.hpp"

using namespace std;

namespace {

inline int randomBetween(int minValue, int maxValue)
{
    int d = maxValue - minValue;
    BOOST_ASSERT(d >= 0);
    return minValue + rand() % (d + 1);
}

inline double randomBetween(double minValue, double maxValue)
{
    int d = maxValue - minValue;
    BOOST_ASSERT(d >= 0);
    if (d == 0)
        return minValue;
    else
        return minValue + d * (static_cast<double>(rand()) / RAND_MAX);
}

class MyAbstractGraphicsShapeItem
{
public:
    virtual ~MyAbstractGraphicsShapeItem() = default;

    virtual void paint(QPainter* painter) const = 0;

    void setPos(const QPointF& pos) {
        m_pos = pos;
    }

    void setPos(qreal x, qreal y) {
        m_pos = {x, y};
    }

    void setRotation(qreal angle) {
        m_rotation = angle;
    }

    void setScale(qreal factor) {
        m_scale = factor;
    }

    void setPen(const QPen& pen) {
        m_pen = pen;
    }

    void setBrush(const QBrush& brush) {
        m_brush = brush;
    }

    QPointF pos() const {
        return m_pos;
    }

    qreal rotation() const {
        return m_rotation;
    }

    qreal scale() const {
        return m_scale;
    }

    QTransform transform() const
    {
        QTransform result;
        result.translate(m_pos.x(), m_pos.y());
        result.rotate(m_rotation);
        result.scale(m_scale, m_scale);
        return result;
    }

    const QPen& pen() const {
        return m_pen;
    }

    const QBrush& brush() const {
        return m_brush;
    }

private:
    QPointF m_pos = {0, 0};
    qreal m_rotation = 0;
    qreal m_scale = 1;
    QPen m_pen;
    QBrush m_brush;
};

class MyGraphicsEllipseItem : public MyAbstractGraphicsShapeItem
{
public:
    MyGraphicsEllipseItem(qreal x, qreal y, qreal width, qreal height) :
      m_x(x),
      m_y(y),
      m_width(width),
      m_height(height)
    {}

    void paint(QPainter* painter) const override {
        painter->drawEllipse(m_x, m_y, m_width, m_height);
    }

private:
    qreal m_x;
    qreal m_y;
    qreal m_width;
    qreal m_height;
};

class MyGraphicsRectItem : public MyAbstractGraphicsShapeItem
{
public:
    MyGraphicsRectItem(qreal x, qreal y, qreal width, qreal height) :
      m_x(x),
      m_y(y),
      m_width(width),
      m_height(height)
    {}

    void paint(QPainter* painter) const override {
        painter->drawRect(m_x, m_y, m_width, m_height);
    }

private:
    qreal m_x;
    qreal m_y;
    qreal m_width;
    qreal m_height;
};

class MyGraphicsPolygonItem : public MyAbstractGraphicsShapeItem
{
public:
    explicit MyGraphicsPolygonItem(const QPolygonF &polygon) :
      m_polygon(polygon)
    {}

    void paint(QPainter* painter) const override {
        painter->drawPolygon(m_polygon);
    }

private:
    QPolygonF m_polygon;
};

class MyGraphicsScene
{
public:
    explicit MyGraphicsScene(const QRectF&) {}

    ~MyGraphicsScene()
    {
        for (auto item : m_items)
            delete item;
    }

    void render(QPainter *painter, const QRectF&)
    {
        for (auto item : m_items) {
            painter->save();
            painter->setTransform(item->transform());
            painter->setPen(item->pen());
            painter->setBrush(item->brush());
            item->paint(painter);
            painter->restore();
        }
    }

    void addItem(MyAbstractGraphicsShapeItem *item) {
        m_items.push_back(item);
    }

    void removeItem(MyAbstractGraphicsShapeItem *item)
    {
        auto it = find(m_items.begin(), m_items.end(), item);
        BOOST_ASSERT(it != m_items.end());
        m_items.erase(it);
    }

private:
    vector<MyAbstractGraphicsShapeItem*> m_items;
};

} // anonymous namespace

struct AnimatedScene::Impl
{
public:
    Impl(int width, int height) :
      m_background(width, height, QImage::Format_RGB32),
      m_traces(width, height, QImage::Format_ARGB32),
      m_scene(QRectF(0, 0, width, height))
    {
        m_background.fill(0);
        addNoise(10000);
        m_traces.fill(0);
        for (auto i=0; i<20; ++i)
            addRandomShape();
    }

    void advance()
    {
        // addNoise(10);
        computeNextState();
    }

    QImage render()
    {
        // Fade m_traces out
        {
            auto b = m_traces.bits();
            size_t size = m_traces.width() * m_traces.height() * 4;
            for (size_t i=3; i<size; i+=4)
                b[i] = static_cast<unsigned char>(static_cast<unsigned int>(b[i])*99/100);
        }

        // Render scene over m_traces
        {
            QPainter painter(&m_traces);
            painter.setRenderHint(QPainter::Antialiasing);
            m_scene.render(&painter, m_traces.rect());
        }

        // Blend scene and traces over the background with stars
        QImage result = m_background;
        {
            QPainter painter(&result);
            painter.drawImage(0, 0, m_traces);
        }
        return result;
    }

private:
    QImage m_background;
    QImage m_traces;
    MyGraphicsScene m_scene;

    struct ItemData
    {
        MyAbstractGraphicsShapeItem *item = nullptr;
        s3dmm::Vec2<double> speed = {0, 0};
        double mass = 1;
        double size = 1;
        double angularSpeed = 0;
    };
    vector<ItemData> m_itemData;

    QSize sceneSize() const {
        return m_background.size();
    }

    void addNoise(unsigned int pixelCount)
    {
        auto size = sceneSize();
        auto h = size.height();
        auto w = size.width();
        auto d = m_background.bits();
        for (auto i=0u; i<pixelCount; ++i) {
            auto x = rand() % w;
            auto y = rand() % h;
            auto r = rand() % 0xff;
            auto g = rand() % 0xff;
            auto b = rand() % 0xff;
            auto pixel = d + ((y*w + x) << 2);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
            pixel[3] = 0xff;
        }
    }

    void addRandomShape()
    {
        enum {
            Ellipse,
            Rect,
            Polygon,
            ShapeTypeCount
        };
        auto size = sceneSize();
        auto shapeSize = min(size.width(), size.height()) / 20;
        auto item = [&]() -> MyAbstractGraphicsShapeItem* {
            switch(rand() % ShapeTypeCount) {
                case Ellipse:
                    return makeRandomEllipse(shapeSize);
                case Rect:
                    return makeRandomRect(shapeSize);
                case Polygon:
                    return makeRandomPolygon(shapeSize);
                default:
                    BOOST_ASSERT(false);
                    return nullptr;
            }
        }();
        auto xpos = shapeSize/2 + randomBetween(0, sceneSize().width() - shapeSize);
        auto ypos = shapeSize/2 + randomBetween(0, sceneSize().height() - shapeSize);
        item->setPos(xpos, ypos);
        item->setRotation(randomBetween(0, 360));
        item->setPen(QPen(randomColor(), randomBetween(0, 5)));
        item->setBrush(randomColor());
        m_scene.addItem(item);
        m_itemData.push_back({
            item,
            randomSpeed(),
            randomBetween(1., 10.),             // mass
            randomBetween(0.5, 2.),             // size
            randomBetween(-M_PI*10, M_PI*10)    // angular speed
        });
    }

    static MyAbstractGraphicsShapeItem *makeRandomEllipse(int maxSize)
    {
        auto minSize = maxSize / 3;
        auto width = randomBetween(minSize, maxSize);
        auto height = randomBetween(minSize, maxSize);
        return new MyGraphicsEllipseItem(-width/2, -height/2, width, height);
    }

    static MyAbstractGraphicsShapeItem *makeRandomRect(int maxSize)
    {
        auto minSize = maxSize / 3;
        auto width = randomBetween(minSize, maxSize);
        auto height = randomBetween(minSize, maxSize);
        return new MyGraphicsRectItem(-width/2, -height/2, width, height);
    }

    static MyAbstractGraphicsShapeItem *makeRandomPolygon(int maxSize)
    {
        auto rmax = maxSize / 2;
        auto rmin = rmax / 5;
        auto vertexCount = randomBetween(3, 10);
        QPolygonF poly;
        for (auto i=0; i<vertexCount; ++i) {
            auto angle = 2*M_PI*i/vertexCount;
            auto r = randomBetween(rmin, rmax);
            auto x = r*cos(angle);
            auto y = r*sin(angle);
            poly << QPointF(x, y);
        }
        return new MyGraphicsPolygonItem(poly);
    }

    static QColor randomColor() {
        return QColor::fromHsv(randomBetween(0, 359), randomBetween(0, 255), randomBetween(150, 255));
    }

    static s3dmm::Vec2<double> randomSpeed()
    {
        auto magnitude = randomBetween(1., 10.);
        auto angle = randomBetween(0., 2*M_PI);
        return {
            magnitude*cos(angle),
            magnitude*sin(angle)
        };
    }

    void computeNextState()
    {
        auto x = getState();
        auto h = 0.1;
        auto k1 = rhs(x);
        auto buf = x;
        axpy(h/2, k1, buf);
        auto k2 = rhs(buf);
        buf = x;
        axpy(h/2, k2, buf);
        auto k3 = rhs(buf);
        buf = x;
        axpy(h, k3, buf);
        auto k4 = rhs(buf);
        for (size_t i=0, n=x.size(); i<n; ++i)
            x[i] += h/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        setState(x);
        processCollisionsWithBorders();
        controlPopulation();
    }

    void processCollisionsWithBorders()
    {
        auto processWallCollision = [](double x, double xmin, double xmax, double& v)
        {
            if ((x < xmin && v < 0) || (x > xmax && v > 0))
                v = -v;
        };
        auto w = m_background.width();
        auto h = m_background.height();
        for (auto& idata : m_itemData) {
            auto pos = idata.item->pos();
            processWallCollision(pos.x(), 0, w, idata.speed[0]);
            processWallCollision(pos.y(), 0, h, idata.speed[1]);
        }
    }

    void controlPopulation()
    {
        auto sizeThreshold = 0.1;
        size_t removedItemCount = 0;
        for (size_t i=0, n=m_itemData.size(); i<n; ++i) {
            auto& idata = m_itemData[i];
            if (idata.size < sizeThreshold) {
                m_scene.removeItem(idata.item);
                delete idata.item;
                m_itemData.erase(m_itemData.begin() + i);
                --i;
                --n;
                ++removedItemCount;
            }
        }

        for (size_t i=0; i<removedItemCount; ++i)
            addRandomShape();
    }

    static constexpr size_t ItemStateSize = 4; // x, y, angle, size
    size_t configurationSize() const {
        return ItemStateSize * m_itemData.size();
    }

    vector<double> getState() const
    {
        auto csize = configurationSize();
        auto ssize = 2 * csize;
        vector<double> result(ssize);
        size_t ix = 0;
        size_t iv = csize;
        for(auto& idata : m_itemData) {
            auto pos = idata.item->pos();
            result[ix++] = pos.x();
            result[ix++] = pos.y();
            result[ix++] = idata.item->rotation();
            result[ix++] = idata.size;
            result[iv++] = idata.speed[0];
            result[iv++] = idata.speed[1];
            result[iv++] = idata.angularSpeed;
            result[iv++] = 0;   // Size speed
        }
        return result;
    }

    void setState(const vector<double>& state)
    {
        auto csize = configurationSize();
        auto ssize = 2 * csize;
        BOOST_ASSERT(state.size() == ssize);
        size_t ix = 0;
        size_t iv = csize;
        for(auto& idata : m_itemData) {
            QPointF pos;
            pos.setX(state[ix++]);
            pos.setY(state[ix++]);
            idata.item->setPos(pos);
            idata.item->setRotation(state[ix++]);
            idata.size = state[ix++];
            idata.item->setScale(idata.size);
            idata.speed[0] = state[iv++];
            idata.speed[1] = state[iv++];
            idata.angularSpeed = state[iv++];
            iv++;   // Unused size speed
        }
    }

    vector<double> rhs(const vector<double>& state) const
    {
        auto csize = configurationSize();
        auto ssize = 2 * csize;
        BOOST_ASSERT(state.size() == ssize);
        vector<double> result(ssize);

        // Compute speeds
        for (size_t ix=0; ix<csize; ++ix)
            result[ix] = state[csize + ix];

        // Compute accelerations
        fill(result.begin()+csize, result.end(), 0);

        // Repulsive forces between particles
        constexpr auto G = 1000.;
        auto n = m_itemData.size();
        for (size_t i1=0; i1+1<n; ++i1) {
            auto i1state = i1*ItemStateSize;
            s3dmm::Vec2<double> r1 = {state[i1state+0], state[i1state+1]};
            for (size_t i2=i1+1; i2<n; ++i2) {
                auto i2state = i2*ItemStateSize;
                s3dmm::Vec2<double> r2 = {state[i2state+0], state[i2state+1]};
                auto r12 = r2 - r1;
                auto d2 = norm2(r12);
                auto d = sqrt(d2);
                if (d <= 0)
                    continue;
                auto e = r12 / d;
                auto f = e * (m_itemData[i1].mass * m_itemData[i2].mass * G / d2);
                result[i1state+csize+0] -= f[0];
                result[i1state+csize+1] -= f[1];
                result[i2state+csize+0] += f[0];
                result[i2state+csize+1] += f[1];
            }
        }

        // Size dynamics
        if (m_itemData.size() > 1) {
            auto size = sceneSize();
            auto distThreshold = 0.25 * min(size.width(), size.height());
            for (size_t i1=0; i1<n; ++i1) {
                auto i1state = i1*ItemStateSize;
                auto dmin = -1.;
                s3dmm::Vec2<double> r1 = {state[i1state+0], state[i1state+1]};
                for (size_t i2=0; i2<n; ++i2) {
                    if (i2 == i1)
                        continue;
                    auto i2state = i2*ItemStateSize;
                    s3dmm::Vec2<double> r2 = {state[i2state+0], state[i2state+1]};
                    auto r12 = r2 - r1;
                    auto d2 = norm2(r12);
                    if (dmin < 0 || dmin > d2)
                        dmin = d2;
                }
                dmin = sqrt(dmin) / distThreshold;
                result[i1state+3] = 0.1*log(dmin)*state[i1state+3];
            }
        }

        return result;
    }

    static void axpy(double a, const vector<double>& x, vector<double>& y)
    {
        auto n = x.size();
        BOOST_ASSERT(n == y.size());
        for (size_t i=0; i<n; ++i)
            y[i] += a*x[i];
    }
};



AnimatedScene::AnimatedScene(int width, int height) :
    m_impl(make_unique<Impl>(width, height))
{
}

AnimatedScene::~AnimatedScene() = default;

void AnimatedScene::advance() {
    m_impl->advance();
}

QImage AnimatedScene::render() {
    return m_impl->render();
}
