#include "s3dmm/Metadata.hpp"

#include <QApplication>
#include <QImage>
#include <QPainter>

#include <fstream>
#include <regex>

#include <boost/program_options.hpp>

using namespace std;
using namespace s3dmm;

using MD = Metadata<2>;
using BT = MD::BT;
using BlockIndex = BT::BlockIndex;
using BlockId = BT::BlockId;

std::pair<unsigned int, BlockIndex> parseBlockIndex(const string& s)
{
    if (s.empty())
        return { 0, BlockIndex() };

    // Parse output subtree id
    regex rx("^(\\d+),(\\d+),(\\d+)$");
    smatch m;
    if (!regex_match(s, m, rx))
        throw runtime_error("Invalid subtree block id");
    BOOST_ASSERT(m.size() == 4);
    unsigned int n[3];
    transform(m.begin()+1, m.end(), n, [](const auto& x) {
        return boost::lexical_cast<unsigned int>(x);
    });
    return { n[0], {n[1], n[2]} };
}

struct Param
{
    string metadataFileName;
    int imageSize = 1024;
    bool displayIds = true;
    unsigned int maxDepth = 0;
    string outputImageFileName;
    string root;
    bool compact = false;
};

void run(const Param& param)
{
    ifstream is(param.metadataFileName);
    if (is.fail())
        throw runtime_error(string("Failed to open input metadata file '") + param.metadataFileName + "'");

    auto rootPos = parseBlockIndex(param.root);
    MD m(is);
    auto& bt = m.blockTree();
    auto root = bt.blockAt(rootPos.second, rootPos.first);
    auto maxDepth = param.maxDepth? param.maxDepth: ~0u;

    QImage img(param.imageSize, param.imageSize, QImage::Format_RGB32);
    img.fill(QRgb(0xffffff));
    QPainter painter(&img);
    painter.setRenderHint(QPainter::Antialiasing);
    vector<QRectF> nodeRects;
    nodeRects.push_back(QRectF(0, 0, param.imageSize, param.imageSize));
    auto drawNodes = [&](auto drawNode) {
        bt.walk(root, maxDepth, [&](const BlockId& blockId) {
            auto relLevel = blockId.level - root.level;
            BOOST_ASSERT(relLevel <= nodeRects.size());
            const auto relMargin = param.compact? 0.0: 0.02;
            const auto normalRelMidSpace = param.compact? 0.1: 0.05;
            const auto relMidspace = param.compact? 0.0: normalRelMidSpace;
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
            drawNode(blockId, rect, normalRelMidSpace);
        });
    };
    auto drawNodeRect = [&](const QRectF& rect) {
        painter.setPen(QColor::fromRgba(0x20000000));
        painter.setBrush(QColor::fromRgba(0x10000000));
        painter.drawRect(rect);
    };
    auto drawId = [&](const BlockId& blockId, const QRectF& rect, double normalRelMidSpace) {
        auto pixelSize = static_cast<int>(normalRelMidSpace*rect.width() + 0.5);
        const int MinFontPixelSize = 4;
        const int QuiteSmallFontPixelSize = 15;
        if (pixelSize >= MinFontPixelSize) {
            if (param.compact) {
                painter.setBrush(Qt::white);
                painter.setPen(pixelSize <= QuiteSmallFontPixelSize? Qt::white: QColor::fromRgba(0x40000000));
                auto r = pixelSize * 0.7;
                painter.drawEllipse(rect.center(), r, r);
            }
            QFont font;
            font.setPixelSize(pixelSize);
            painter.setFont(font);
            painter.setPen(Qt::black);
            painter.drawText(rect, Qt::AlignCenter, QString::number(blockId.index));
        }
    };

    if (param.compact && param.displayIds) {
        drawNodes([&](const BlockId&, const QRectF& rect, double) {
            drawNodeRect(rect);
        });
        drawNodes(drawId);
    }
    else {
        drawNodes([&](const BlockId& blockId, const QRectF& rect, double normalRelMidSpace) {
            drawNodeRect(rect);
            if (param.displayIds)
                drawId(blockId, rect, normalRelMidSpace);
        });
    }
    auto outputImageFileName = param.outputImageFileName.empty()? param.metadataFileName + ".png": param.outputImageFileName;
    img.save(outputImageFileName.c_str());
    cout << "Wrote image to file '" << outputImageFileName << "'" << endl;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    try {
        Param param;

        namespace po = boost::program_options;
        auto po_value = [](auto& x) {
            return po::value(&x)->default_value(x);
        };
        po::options_description po_generic("Gerneric options");
        po_generic.add_options()
                ("help,h", "produce help message");

        po::options_description po_main("Main parameters");
        po_main.add_options()
                ("input", po_value(param.metadataFileName), "Metadata file name (must contain 2D metadata)")
                ("size", po_value(param.imageSize), "Output image width and height")
                ("ids", po_value(param.displayIds), "Display node ids")
                ("depth", po_value(param.maxDepth), "Maximal depth")
                ("output", po_value(param.outputImageFileName), "Output image file name (by default, based on input)")
                ("root", po_value(param.root), "Subtree root (by default, the root of the block tree)")
                ("compact", po_value(param.compact), "Compact view");
        po::variables_map vm;
        auto po_alloptions = po::options_description()
                .add(po_generic).add(po_main);
        po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << po_alloptions << "\n";
            return 0;
        }

        run(param);

        return EXIT_SUCCESS;
    }
    catch(exception& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}
