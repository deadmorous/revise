#include "saveFbImage.hpp"

#include <QImage>
#include <QDir>
#include <GL/gl.h>
#include <vector>

using namespace std;

void saveFbImage(int width, int height)
{
    auto size = static_cast<size_t>(3 * width * height);

    vector<unsigned char> pixels(size);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Reverse the order of image lines
    {
        vector<unsigned char> line(static_cast<size_t>(3 * width));
        auto ldata = line.data();
        unsigned char *n1 = pixels.data(),   *n2 = pixels.data() + (height - 1)*width*3;
        for (; n1<n2; n1+=3*width, n2-=3*width) {
            copy(n1, n1+3*width, ldata);
            copy(n2, n2+3*width, n1);
            copy(ldata, ldata+3*width, n2);
        }
    }

    QImage image(pixels.data(), width,  height, QImage::Format_RGB888);

    static unsigned int imageCount = 0;
    ++imageCount;
    auto imageName = QString("image_%1.png").arg(imageCount, 5, 10, QChar('0'));

    auto imageDirName = "images";
    QDir dir;
    if (!dir.exists(imageDirName))
        dir.mkdir(imageDirName);
    dir.cd(imageDirName);
    auto fileName = dir.absoluteFilePath(imageName);
    image.save(fileName);
}
