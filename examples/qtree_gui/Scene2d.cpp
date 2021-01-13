#include "Scene2d.h"
#include "StateDataController.h"
#include "ViewTransformController.h"

#include <QPainter>

using namespace std;
using namespace s3dmm;

namespace {

inline QPointF toQPointF(const Vec<2, real_type>& x) {
    return { static_cast<qreal>(x[0]), static_cast<qreal>(x[1]) };
}

inline Vec<2, real_type> toVec(const QPointF& x) {
    return { static_cast<real_type>(x.x()), static_cast<real_type>(x.y()) };
}

template <class F>
inline void tryPaint(QPaintDevice *dev, const QRect& rect, F f)
{
    try {
        QPainter painter(dev);
        f(painter, rect);
    }
    catch (const std::exception& e) {
        auto text = QString::fromUtf8(e.what());
        QPainter painter(dev);
        painter.fillRect(rect, Qt::white);
        painter.setPen(Qt::red);
        QFont font;
        font.setPixelSize(20);
        painter.setFont(font);
        painter.drawText(rect, Qt::AlignCenter | Qt::TextWordWrap, text);
    }
}

inline QRectF maxSquare(const QRectF& rect)
{
    auto d = rect.width() - rect.height();
    QRectF result;
    if (d > 0)
        result = QRectF(rect.left()+d/2, rect.top(), rect.height(), rect.height());
    else
        result = QRectF(rect.left(), rect.top()-d/2, rect.width(), rect.width());
    constexpr const auto MarginSize = 25.;
    if (result.width() > 2*MarginSize && result.height() > 2*MarginSize)
        result.adjust(MarginSize, MarginSize, -MarginSize, -MarginSize);
    return result;
}

struct PaintHelper
{
    PaintHelper(QPainter& painter, const QRect& rect, const StateData& sd)
    {
        auto& md = sd.metadata();
        auto& bt = md.blockTree();
        auto rootLocation = sd.blockTreeLocation();
        root = bt.blockAt(rootLocation.index, rootLocation.level);
        rootPos = bt.blockPos(root);

        sqrect = maxSquare(rect);
        auto rootCenter = rootPos.center().convertTo<qreal>();
        modelTransform.translate(sqrect.center().x(), sqrect.center().y());
        scale = sqrect.width() / static_cast<double>(rootPos.size());
        modelTransform.scale(scale, -scale);
        modelTransform.translate(-rootCenter[0], -rootCenter[1]);

        painterTransform = painter.transform();
        ipainterTransform = painterTransform.inverted();
        painter.setTransform(QTransform());
        pixelSize = ipainterTransform.mapRect(QRectF(0,0,1,1)).width();

        fullTransform = modelTransform * painterTransform;

        rectf = rect;
    }
    StateData::Metadata::BT::BlockId root;
    BoundingCube<2, real_type> rootPos;
    QRectF sqrect;
    QTransform modelTransform;
    double scale;
    QTransform painterTransform;
    QTransform ipainterTransform;
    QTransform fullTransform;
    double pixelSize;
    QRectF rectf;
};

void paintQtree(QPainter& painter, const QRect& rect, const StateData& sd)
{
    using BlockId = StateData::Metadata::BT::BlockId;

    PaintHelper ph(painter, rect, sd);

    auto compact = sd.qtreeCompactView();
    auto displayIds = sd.qtreeDisplayIds();

    vector<QRectF> nodeRects;
    auto size = ph.sqrect.width();

    auto widgetRect = ph.ipainterTransform.mapRect(QRectF(rect));
    auto minRectSize = ph.pixelSize;

    nodeRects.push_back(QRectF(ph.sqrect.left(), ph.sqrect.top(), size, size));
    auto& bt = sd.metadata().blockTree();
    auto hasField = sd.hasSparseField();
    vector<real_type> trash;
    auto& sf = hasField? sd.sparseField(): trash;
    auto& btn = sd.rootSubtreeNodes();
    QPen nodeRectPen(QColor::fromRgba(0x20000000));
    // nodeRectPen.setCosmetic(true);
    QPen nodeOutlinePen;
    // nodeOutlinePen.setCosmetic(true);
    nodeOutlinePen.setJoinStyle(Qt::MiterJoin);
    QPen nodeOuterOutlinePen(QColor::fromRgb(sd.displaySparseNodes()? 0xaaaaaa: 0x000000), 2);
    // nodeOuterOutlinePen.setCosmetic(true);
    nodeOuterOutlinePen.setJoinStyle(Qt::MiterJoin);
    QBrush nodeRectBrush(QColor::fromRgba(hasField? 0x00000000: 0x10000000));
    QBrush nodeRectWithFieldBrush(QColor::fromRgba(0x10880000));
    auto maxDepth = sd.limitedQtree()? btn.maxDepth(): ~0u;
    auto filled = sd.quadtreeNodefill();
    auto drawNodes = [&](auto drawNode) {
        bt.walk(ph.root, [&](const BlockId& blockId) {
            auto relLevel = blockId.level - ph.root.level;
            BOOST_ASSERT(relLevel <= nodeRects.size());
            const auto relMargin = compact? 0.0: 0.02;
            const auto normalRelMidSpace = compact? 0.1: 0.05;
            const auto relMidspace = compact? 0.0: normalRelMidSpace;
            QRectF rect;
            if (relLevel == 0)
                rect = nodeRects.front();
            else {
                nodeRects.resize(relLevel);
                auto parentRect = nodeRects.back();
                auto parentSize = parentRect.width();
                const auto margin = relMargin*parentSize;
                const auto midspace = relMidspace*parentSize;
                auto loc = blockId.location & 1;
                auto size = 0.5*(parentRect.width() - 2*margin - midspace);
                rect.setLeft(loc[0] == 0? parentRect.left() + margin: parentRect.right()-margin-size);
                rect.setTop(loc[1] == 0? parentRect.bottom()-margin-size: parentRect.top()+margin);
                rect.setWidth(size);
                rect.setHeight(size);
                nodeRects.push_back(rect);
            }
            if (rect.width() < minRectSize || !widgetRect.intersects(rect))
                return false;
            drawNode(blockId, rect, normalRelMidSpace);
            return relLevel < maxDepth;
        });
    };
    auto blockHasField = [&](const BlockId& blockId) {
        if (hasField) {
            if (blockId.level - ph.root.level > btn.maxDepth())
                return false;
            auto result = true;
            btn.walkBlockNodes(blockId, [&](
                               unsigned int localNodeNumber,
                               const auto&,
                               std::size_t nodeNumber)
            {
                if (sf[nodeNumber] == StateData::BlockTreeFieldProvider::noFieldValue())
                    result = false;
            });
            return result;
        }
        else
            return false;
    };
    auto drawNodeRect = [&](const BlockId& blockId, const QRectF& rect) {
        painter.setPen(nodeRectPen);
        painter.setBrush(filled? (blockHasField(blockId)? nodeRectWithFieldBrush: nodeRectBrush): Qt::NoBrush);
        painter.drawRect(ph.painterTransform.mapRect(rect));
    };
    auto drawNodeOutline = [&](const BlockId& blockId, const QRectF& rect) {
        auto halfWidth = static_cast<int>(blockId.level);
        nodeOutlinePen.setWidth(halfWidth << 1);
        nodeOutlinePen.setColor(QColor::fromRgb(blockHasField(blockId)? 0xbb8888: 0x888888));
        painter.setPen(nodeOutlinePen);
        painter.setBrush(Qt::NoBrush);
        auto rc = ph.painterTransform.mapRect(rect);
        if (!sd.displaySparseNodes())
            painter.drawRect(rc.adjusted(halfWidth, halfWidth, -halfWidth, -halfWidth));
        painter.setPen(nodeOuterOutlinePen);
        painter.drawRect(rc);
    };
    QFont idFont;
    QPen idPen(Qt::black);
    QBrush idPadBrush(QColor::fromRgba(sd.fill()? 0x88ffffff: 0xffffffff));
    QPen idPadNormalPen(QColor::fromRgba(0x40000000));
    // idPadNormalPen.setCosmetic(true);
    QPen idPadNoPen(Qt::NoPen);
    const double MinFontPixelSize = 4;
    const double QuiteSmallFontPixelSize = 15;
    auto drawId = [&](const BlockId& blockId, const QRectF& rect, double normalRelMidSpace) {
        auto fontPixelSize = static_cast<int>(normalRelMidSpace*rect.width()/ph.pixelSize + 0.5);
        if (fontPixelSize >= MinFontPixelSize) {
            if (compact) {
                painter.setBrush(idPadBrush);
                painter.setPen(fontPixelSize <= QuiteSmallFontPixelSize? idPadNoPen: idPadNormalPen);
                auto r2 = fontPixelSize * 0.7;
                auto r1 = r2 * std::max(1, static_cast<int>(log10(blockId.index + 0.01)));
                painter.drawEllipse(ph.painterTransform.map(rect.center()), r1, r2);
            }
            idFont.setPixelSize(fontPixelSize);
            painter.setFont(idFont);
            painter.setPen(idPen);
            painter.drawText(ph.painterTransform.mapRect(rect), Qt::AlignCenter, QString::number(blockId.index));
        }
    };

    if (compact && displayIds) {
        painter.setRenderHint(QPainter::Antialiasing, false);
        if (sd.fill())
            drawNodes([&](const BlockId& blockId, const QRectF& rect, double) {
                drawNodeRect(blockId, rect);
            });
        else
            drawNodes([&](const BlockId& blockId, const QRectF& rect, double) {
                drawNodeOutline(blockId, rect);
            });
        painter.setRenderHint(QPainter::Antialiasing, true);
        drawNodes(drawId);
    }
    else {
        drawNodes([&](const BlockId& blockId, const QRectF& rect, double normalRelMidSpace) {
            painter.setRenderHint(QPainter::Antialiasing, false);
            if (sd.fill())
                drawNodeRect(blockId, rect);
            else
                drawNodeOutline(blockId, rect);
            if (displayIds) {
                painter.setRenderHint(QPainter::Antialiasing, true);
                drawId(blockId, rect, normalRelMidSpace);
            }
        });
    }
}

void paintMesh(QPainter& painter, const QRect& rect, const StateData& sd)
{
    PaintHelper ph(painter, rect, sd);
    auto& mesh = sd.mesh();
    auto cache = mesh.makeCache();
    auto vars = mesh.coordinateVariables();
    QPolygonF polygon;
    QPen pen;
    // pen.setCosmetic(true);
    if (sd.fill()) {
        painter.setBrush(QColor::fromRgba(0x2200cc00));
        pen.setColor(QColor::fromRgba(0x44008800));
    }
    else {
        // painter.setBrush(Qt::NoBrush);
        pen.setColor(QColor::fromRgba(0xffcc0000));
        pen.setWidth(3);
        pen.setJoinStyle(Qt::MiterJoin);
        QBrush brush(pen.color());
        // brush.setStyle(Qt::BDiagPattern);
        brush.setStyle(Qt::Dense7Pattern);
        painter.setBrush(brush);
        painter.setRenderHint(QPainter::Antialiasing);
    }
    painter.setPen(pen);
    for (auto zone : mesh.zones(cache, vars)) {
        auto elements = zone.elements();
        polygon.resize(static_cast<int>(zone.nodesPerElement()));
        for (auto it=elements.begin(); it!=elements.end(); ++it) {
            auto elementNodes = it.elementNodes();
            boost::range::transform(elementNodes, polygon.begin(), [&](const auto& nodePos) {
                return ph.fullTransform.map(QPointF(static_cast<qreal>(nodePos[0]), static_cast<qreal>(nodePos[1])));
            });
            if (std::any_of(polygon.begin(), polygon.end(), [&](const auto& x) { return ph.rectf.contains(x); }))
                painter.drawPolygon(polygon);
        }
    }
}

void paintBoundary(QPainter& painter, const QRect& rect, const StateData& sd)
{
    PaintHelper ph(painter, rect, sd);

    painter.setRenderHint(QPainter::Antialiasing);

    auto& boundary = sd.boundary();

    // Draw boundary edges and compute average edge length
    QPen boundaryPen = sd.fill()? QPen(QColor::fromRgba(0x880000aa), 2): QPen(QColor::fromRgba(0xff2222cc), 5);
    // boundaryPen.setCosmetic(true);
    painter.setPen(boundaryPen);
    double avgEdgeLength = 0;
    auto edgeCount = 0u;
    for (auto& zone : boundary.zones()) {
        if (zone.elementType() != MeshElementType::Quad)
            throw runtime_error("paintBoundary() failed: Only quads are currently supported");
        auto& zoneData = zone.data<MeshElementType::Quad>();
        auto& zoneNodes = zoneData.nodes;
        for (auto& fd : zoneData.faces) {
            auto& face = fd.face;
            auto n1 = zoneNodes[face[0]];
            auto n2 = zoneNodes[face[1]];
            auto x1 = ph.fullTransform.map(toQPointF(n1));
            auto x2 = ph.fullTransform.map(toQPointF(n2));
            if (ph.rectf.contains(x1) || ph.rectf.contains(x2)) {
                painter.drawLine(x1, x2);
                auto edgeLength = norm((n2-n1).convertTo<double>());
                avgEdgeLength += edgeLength;
                ++edgeCount;
                constexpr const double NodeSpacingThresholdForDirMarkers = 15;
                constexpr const double DirMarkerPixelSize = 2;
                if (sd.displayBoundaryDirMarkers() && edgeLength*ph.scale/ph.pixelSize >= NodeSpacingThresholdForDirMarkers) {
                    auto x0 = toVec(x1*0.8 + x2*0.2);
                    auto d = rot_ccw(toVec(x2-x1)) * static_cast<real_type>(DirMarkerPixelSize*ph.pixelSize/(ph.scale*edgeLength));
                    painter.drawLine(toQPointF(x0-d), toQPointF(x0+d));
                }
            }
        }
    }
    if (edgeCount > 0)
        avgEdgeLength /= edgeCount;

    painter.setBrush(QColor::fromRgba(0xcc0000cc));
    painter.setPen(Qt::NoPen);
    auto nodeRadius = 2;
    for (auto& zone : boundary.zones()) {
        auto& zoneData = zone.data<MeshElementType::Quad>();
        auto& zoneNodes = zoneData.nodes;
        for (auto& node : zoneNodes) {
            auto x = ph.fullTransform.map(toQPointF(node));
            if (ph.rectf.contains(x))
                painter.drawEllipse(x, nodeRadius, nodeRadius);
        }
    }

    // Draw boundary node numbers
    if (sd.displayBoundaryNodeNumbers()) {
        constexpr const double NodeSpacingThresholdForLabels = 50;
        painter.setBrush(QColor::fromRgb(0xccccff));
        QPen textPen(Qt::black);
        QPen textRectPen(QColor::fromRgb(0x888888));
        if (avgEdgeLength*ph.scale/ph.pixelSize >= NodeSpacingThresholdForLabels) {
            for (auto& zone : boundary.zones()) {
                auto& zoneData = zone.data<MeshElementType::Quad>();
                auto& zoneNodes = zoneData.nodes;
                auto inode = 0;
                for (auto& node : zoneNodes) {
                    auto x = ph.fullTransform.map(toQPointF(node));
                    if (ph.rectf.contains(x)) {
                        auto text = QString::number(inode);
                        auto textRectSize = painter.boundingRect(ph.rectf, text).size() + QSizeF(2,2);
                        QRectF textRect(x+QPointF(2,2), textRectSize);
                        painter.setPen(textRectPen);
                        painter.drawRoundedRect(textRect, 2, 2);
                        painter.setPen(textPen);
                        painter.drawText(textRect, Qt::AlignCenter, text);
                    }
                    ++inode;
                }
            }
        }
    }
}

void paintSparseNodes(QPainter& painter, const QRect& rect, const StateData& sd, boost::any& cache)
{
    PaintHelper ph(painter, rect, sd);
    auto& btn = sd.rootSubtreeNodes();
    auto nodeRadius = sd.fill()? 3: 5;
    auto minCellSize = static_cast<double>(ph.sqrect.width()) / (1 << btn.maxDepth());
    auto left = ph.sqrect.left();
    auto bottom = ph.sqrect.bottom();

    // Draw boundary nodes
    auto& d = btn.data();
    painter.setRenderHint(QPainter::Antialiasing);
    if (sd.hasSparseField()) {
        auto& sf = sd.sparseField();
        auto frange = sd.fieldRange();
        QPen novalPen(Qt::black);
        QBrush novalBrush(Qt::white);
        QPen noPen(Qt::NoPen);
        foreach_byindex32(nodeNumber, d.n2i) {
            auto& nodeIndex = d.n2i[nodeNumber];
            auto x = ph.painterTransform.map(QPointF(left+nodeIndex[0]*minCellSize, bottom-nodeIndex[1]*minCellSize));
            if (ph.rectf.contains(x)) {
                if (sf[nodeNumber] == StateData::BlockTreeFieldProvider::noFieldValue()) {
                    painter.setPen(novalPen);
                    painter.setBrush(novalBrush);
                }
                else {
                    auto v = static_cast<double>((sf[nodeNumber] - frange[0]) / (frange[1] - frange[0]));
                    painter.setPen(noPen);
                    painter.setBrush(QColor::fromHsvF(v*5./6, 1, 0.8));
                }
                painter.drawEllipse(x, nodeRadius, nodeRadius);
            }
        }
    }
    else {
        painter.setBrush(Qt::black);
        painter.setPen(Qt::NoPen);
        foreach_byindex32(nodeNumber, d.n2i) {
            auto& nodeIndex = d.n2i[nodeNumber];
            auto x = ph.painterTransform.map(QPointF(left+nodeIndex[0]*minCellSize, bottom-nodeIndex[1]*minCellSize));
            if (ph.rectf.contains(x)) {
                painter.drawEllipse(x, nodeRadius, nodeRadius);
            }
        }
    }

    if (sd.displaySparseNodeNumbers()) {
        if (cache.empty()) {
            // TODO: Reset cache somehow...
            std::vector<MultiIndex<4, unsigned int>> nbrDepth(btn.nodeCount());
            sd.metadata().blockTree().walk(btn.root(), btn.maxDepth(), [&](auto& blockId) {
                btn.walkBlockNodes(
                            blockId, [&](
                            unsigned int localNodeNumber,
                            const auto&,
                            std::size_t nodeNumber)
                {
                    auto& n = nbrDepth[nodeNumber][localNodeNumber];
                    auto relLevel = blockId.level - ph.root.level;
                    if (n < relLevel)
                        n = relLevel;
                });
            });
            // lowest 2 bits is the local node number, highest bits are max block depth
            std::vector<unsigned int> nbrMaxDepth(btn.nodeCount(), ~0u);
            boost::range::transform(nbrDepth, nbrMaxDepth.begin(), [](const MultiIndex<4, unsigned int>& depths) {
                auto it = std::max_element(depths.begin(), depths.end());
                return (*it) << 2 | (it - depths.begin());
            });
            cache = nbrMaxDepth;
        }

        painter.setBrush(QColor::fromRgb(0x444444));
        QPen textPen(Qt::white);
        QPen textRectPen(Qt::NoPen);
        auto& nbrDepth = boost::any_cast<std::vector<unsigned int>&>(cache);
        foreach_byindex32(nodeNumber, d.n2i) {
            auto& nodeIndex = d.n2i[nodeNumber];
            auto x = ph.painterTransform.map(QPointF(left+nodeIndex[0]*minCellSize, bottom-nodeIndex[1]*minCellSize));
            if (ph.rectf.contains(x)) {
                auto depthData = nbrDepth[nodeNumber];
                auto depth = depthData >> 2;
                auto cellPixelSize = static_cast<double>(ph.sqrect.width()) / ((1 << depth) * ph.pixelSize);
                const auto CellSizeThresholdForNumber = 50;
                const auto TextRectOffset = 2;
                if (cellPixelSize >= CellSizeThresholdForNumber) {
                    auto text = QString::number(nodeNumber);
                    auto textRectSize = painter.boundingRect(ph.rectf, text).size() + QSizeF(2,2);
                    QRectF textRect;
                    switch (depthData & 3) {
                    case 0:
                        textRect = QRectF(
                                    x+QPointF(TextRectOffset,
                                              TextRectOffset),
                                    textRectSize);
                        break;
                    case 1:
                        textRect = QRectF(
                                    x+QPointF(-TextRectOffset-textRectSize.width(),
                                              TextRectOffset),
                                    textRectSize);
                        break;
                    case 2:
                        textRect = QRectF(
                                    x+QPointF(TextRectOffset,
                                              -TextRectOffset-textRectSize.height()),
                                    textRectSize);
                        break;
                    case 3:
                        textRect = QRectF(
                                    x+QPointF(-TextRectOffset-textRectSize.width(),
                                              -TextRectOffset-textRectSize.height()),
                                    textRectSize);
                        break;
                    }
                    painter.setPen(textRectPen);
                    painter.drawRoundedRect(textRect, 2, 2);
                    painter.setPen(textPen);
                    painter.drawText(textRect, Qt::AlignCenter, text);
                }
            }
        }
    }
}

void paintDenseField(QPainter& painter, const QRect& rect, const StateData& sd)
{
    auto& denseField = sd.denseField();
    auto depth = sd.rootSubtreeDepth();
    PaintHelper ph(painter, rect, sd);
    auto n = 1u << depth;
    BOOST_ASSERT(denseField.size() == static_cast<std::size_t>((n+1)*(n+1)));
    auto size = ph.sqrect.width();
    auto cellSize = size / n;
    auto npx = std::max(1, static_cast<int>(cellSize / ph.pixelSize + 0.5));
    auto dt = 1.f / npx;
    auto frange = sd.fieldRange();
    auto texelSize = cellSize * static_cast<double>(dt);
    auto y0 = ph.sqrect.bottom()-0.5*ph.pixelSize;
    for (auto iy=0u; iy<n; ++iy, y0-=cellSize) {
        auto i00 = iy*(n+1);
        auto x0 = ph.sqrect.left()-0.5*ph.pixelSize;
        for (auto ix=0u; ix<n; ++ix, ++i00, x0+=cellSize) {
            auto i10 = i00+1;
            auto i01 = i00+(n+1);
            auto i11 = i01+1;
            if (!(isnan(denseField[i00]) || isnan(denseField[i10]) || isnan(denseField[i01]) || isnan(denseField[i11])) &&
                    ph.rectf.intersects(ph.painterTransform.mapRect(QRectF(QPointF(x0, y0-cellSize), QSizeF(cellSize, cellSize)))))
            {
                auto ty = 0.f;
                for (auto ipy=0; ipy<npx; ++ipy, ty+=dt) {
                    auto tx = 0.f;
                    for (auto ipx=0; ipx<npx; ++ipx, tx+=dt) {
                        auto v =
                                (1-tx)*(denseField[i00]*(1-ty) + denseField[i01]*ty) +
                                tx*(denseField[i10]*(1-ty) + denseField[i11]*ty);
                        v = (v - frange[0]) / (frange[1] - frange[0]);
                        auto c = QColor::fromHsvF(static_cast<double>(v)*5./6, 1, 0.8);
                        c.setAlphaF(0.7);
                        painter.fillRect(
                                    ph.painterTransform.mapRect(QRectF(
                                        QPointF(x0+ipx*texelSize, y0-ipy*texelSize),
                                        QSizeF(texelSize, texelSize))),
                                    c);
                    }
                }
            }
        }
    }
}

} // anonymous namespace

Scene2d::Scene2d(StateDataController *stateDataController, QWidget *parent) :
    QWidget(parent),
    m_stateDataController(stateDataController)
{
    setMinimumSize(800, 600);
    connect(m_stateDataController, &StateDataController::qtreeViewChanged, this, qOverload<>(&QWidget::update));
    m_transformController = new ViewTransformController(this);
    auto resetCache = [this]() { m_sparseNodesCache.clear(); };
    auto& sd = m_stateDataController->stateData();
    sd.onInputFileNameChanged(resetCache);
    sd.onBlockTreeLocationChanged(resetCache);
    setFocusPolicy(Qt::StrongFocus);
}

void Scene2d::paintEvent(QPaintEvent*)
{
    tryPaint(this, rect(), [this](QPainter& painter, const QRect& rect) {
        auto safePaint = [&painter](auto paintFunc) {
            painter.save();
            paintFunc();
            painter.restore();
        };
        painter.fillRect(rect, Qt::white);
        auto& sd = m_stateDataController->stateData();
        m_transformController->transformPainter(painter);
        if (sd.displayQtree())
            safePaint([&]() { paintQtree(painter, rect, sd); });
        if (sd.displayMesh())
            safePaint([&]() { paintMesh(painter, rect, sd); });
        if (sd.displayBoundary())
            safePaint([&]() { paintBoundary(painter, rect, sd); });
        if (sd.displaySparseNodes())
            safePaint([&]() { paintSparseNodes(painter, rect, sd, m_sparseNodesCache); });
        if (sd.displayDenseField())
            safePaint([&]() { paintDenseField(painter, rect, sd); });
    });
}
