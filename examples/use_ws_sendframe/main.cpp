#include <iostream>
#include <thread>
#include <algorithm>

#include "AnimatedScene.hpp"

#include "ws_sendframe/FrameServerThread.hpp"
#include "ws_sendframe/FrameServer.hpp"
#include "s3vs/VsControllerFrameOutputRW.hpp"
#include "ipc/SharedMemory.h"

using namespace std;

void run(int /*argc*/, char* /*argv*/[])
{
    const unsigned int w = 640;
    const unsigned int h = 480;
    using FrameNumberType = std::uint64_t;
    auto shmem = SharedMemory::create(sizeof(s3vs::VsControllerFrameOutputHeader)+w*h*4);
    FrameServerThread frameServerThread(1234);
    FrameServer *frameServer = nullptr;
    for (auto iattempt=0; !frameServer && iattempt<10; ++iattempt) {
        this_thread::sleep_for(chrono::milliseconds(100));
        frameServer = frameServerThread.frameServer();
    }
    if (!frameServer)
        throw runtime_error("Failed to start frame server");

    frameServer->setSourceInfo("a", shmem.shmid(), {w, h});

    AnimatedScene sc(w, h);
    FrameNumberType frameNumber = 0;
    for (auto i=0; i<4500; ++i) {
        sc.advance();
        auto img = sc.render();
        s3vs::VsControllerFrameOutputHeader hdr;
        hdr.frameNumber = frameNumber++;
        s3vs::VsControllerFrameOutputRW rw(&shmem, w*h*4);
        rw.writeFrame(hdr, img.bits());
        this_thread::sleep_for(chrono::milliseconds(40));
    }
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
