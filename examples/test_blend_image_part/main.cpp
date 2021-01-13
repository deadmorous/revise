#include "BlendImagePart.hpp"

#include "s3vs/types.hpp"

#include <exception>
#include <iostream>
#include <sstream>

#include <QImage>

using namespace s3vs;

class ImageMaker
{
public:
    using RGBA4f = std::array<float, 4>;;
    using ARGB4c = std::array<unsigned char, 4>;

    explicit ImageMaker(const Vec2i& size = {0, 0})
    {
        init(size);
    }
    void init(const Vec2i& size)
    {
        m_img.size = size;
        m_img.bits.resize(size[0]*size[1]*4);
        std::fill(m_img.bits.begin(), m_img.bits.end(), 0);
    }
    void setPixel4f(size_t x, size_t y, const RGBA4f& rgba)
    {
        ARGB4c argb2{
            static_cast<unsigned char>(lround(rgba[2] * 255)),
            static_cast<unsigned char>(lround(rgba[1] * 255)),
            static_cast<unsigned char>(lround(rgba[0] * 255)),
            static_cast<unsigned char>(lround(rgba[3] * 255))
        };
        setPixel4c(x, y, argb2);
    }

    void fillImage(std::function<RGBA4f(size_t, size_t)> getRgbaFnc)
    {
        for (auto x = 0; x < m_img.size[0]; x++)
        {
            for (auto y = 0; y < m_img.size[1]; y++)
            {
                auto rgba = getRgbaFnc(x, y);
                setPixel4f(x, y, rgba);
            }
        }
    }
    void fillImage(const RGBA4f& color)
    {
        fillImage([&color](size_t, size_t){
            return color;
        });
    }
    RgbaImage getImage() const
    {
        return m_img;
    }
private:
    RgbaImage m_img;
    ARGB4c m_pixelOutOfRange{0, 0, 0, 0};

    unsigned char* pixelPtr(size_t x, size_t y)
    {
        auto pos = 4*y*m_img.size[0] + 4*x;
        if (pos >= m_img.bits.size())
            return m_pixelOutOfRange.data();
        return m_img.bits.data() + pos;
    }
    void setPixel4c(size_t x, size_t y, const ARGB4c& argb)
    {
        auto pix = pixelPtr(x, y);
        for (auto byte : argb)
        {
            *pix = byte;
            ++pix;
        }
    }
};


using namespace std;


void run(int argc, char* argv[])
{
    Vec2i sizeDst{320, 200};
    ImageMaker im(sizeDst);
    im.fillImage([sizeDst](size_t px, size_t py) {
        ImageMaker::RGBA4f color{0, 0, 0, 0};
        auto c = sizeDst/2.;
        auto r = (std::min(sizeDst[0], sizeDst[1]) - 10) / 2.;
        auto d = sqrt((c[0] - px)*(c[0] - px) + (c[1] - py)*(c[1] - py));
        if (d <= r)
            color[0] = color[3] = 1;
        return color;
    });

    auto imgDst = im.getImage();

    im.init({150, 150});
    im.fillImage({0, 1, 0, 0.5});

    RgbaImagePart imgSrc;
    imgSrc.origin = {20, 20};
    imgSrc.image = im.getImage();

    blendImagePart(imgDst, imgSrc);

    im.fillImage([sz = im.getImage().size](size_t px, size_t py) {
        ImageMaker::RGBA4f color{1, 0, 1, 1};
        color[3] = 1.f*(sz[0] - px)/sz[0] * (sz[1] - py)/sz[1];
        return color;
    });
    imgSrc.origin = {0, 0};
    imgSrc.image = im.getImage();

    blendImagePart(imgDst, imgSrc);

    im.init({50, 50});
    im.fillImage({0, 0, 1, 1});
    imgSrc.origin = {200, 100};
    imgSrc.image = im.getImage();

    blendImagePart(imgDst, imgSrc);

    QImage image(imgDst.bits.data(), imgDst.size[0], imgDst.size[1], QImage::Format_ARGB32);

    if (!image.save("test_blend_image_part.png"))
        cout << "Failed to save image" << endl;
}

int main(int argc, char* argv[])
{
    try
    {
        run(argc, argv);
        return EXIT_SUCCESS;
    }
    catch (exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
