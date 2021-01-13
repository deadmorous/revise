/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

#define _USE_MATH_DEFINES
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <experimental/filesystem>
#include <boost/assert.hpp>
#include "MultiIndex.hpp"
#include "filename_util.hpp"
#include "TecplotMeshWriterHelper.hpp"
#include "BinaryMeshWriterHelper.hpp"

#include <boost/program_options.hpp>

using namespace std;
using namespace s3dmm;

namespace {

real_type deg2rad(real_type deg)
{
    return deg / 180 * M_PI;
}

real_type lerp(real_type x1, real_type x2, unsigned int i, unsigned int n) {
    return x1 + (x2-x1)*i/(n-1);
}

class Refiner
{
public:
    Refiner(real_type f, real_type x0, real_type length, unsigned int n) :
        m_x(x0)
    {
        m_alpha = pow(f, make_real(1)/(n-1));
        if (f == make_real(1))
            m_l = length / n;
        else
            m_l = length * (1-m_alpha) / (1-pow(m_alpha, n));
    }

    real_type value() const {
        return m_x;
    }

    void next()
    {
        m_x += m_l;
        m_l *= m_alpha;
    }

private:
    real_type m_x;
    real_type m_l;
    real_type m_alpha;
};

struct FrameParam
{
    real_type r1 = make_real(0.1);
    real_type r2 = make_real(1);
    real_type theta1 = make_real(10);
    real_type theta2 = make_real(170);
    real_type phi1 = make_real(0);
    real_type phi2 = make_real(270);
    real_type N = make_real(1000);
    real_type sx = make_real(1);
    real_type sy = make_real(1);
    real_type sz = make_real(1);
    real_type fx = make_real(1);    // Refinement in X dimension
    real_type fy = make_real(1);    // Refinement in Y dimension
    real_type fz = make_real(1);    // Refinement in Z dimension
    unsigned int spaceDimension = 3u;
};

struct TimeParam
{
    real_type startTime = make_real(0);
    real_type endTime = make_real(1);
    unsigned int frameCount = 1;
};

template <class WriteHelper>
void generateFrameSphere(WriteHelper& wh, const FrameParam& fp, real_type time)
{
    auto theta1 = deg2rad(fp.theta1);
    auto theta2 = deg2rad(fp.theta2);
    auto phi1 = deg2rad(fp.phi1);
    auto phi2 = deg2rad(fp.phi2);

    auto alpha = fp.r2/fp.r1;
    auto a = (fp.r2-fp.r1)*(alpha-make_real(1))/(alpha+make_real(1));
    auto b = make_real(2)*a/(alpha-make_real(1));

    auto xi = make_real(2)*(fp.r2-fp.r1)/(fp.r2+fp.r1);
    auto dtheta = theta2 - theta1;
    auto dphi = phi2 - phi1;

    auto fieldR = [&time](const real_type& r) {
        auto rt = r*(1+time);
        return static_cast<real_type>(sin(2*M_PI/(5*rt)));
    };
    auto fieldPhi = [&time](const real_type& phi) {
        auto phit = phi*(1+time);
        return sin(phit);
    };
    auto fieldTheta = [&time](const real_type& theta) {
        auto thetat = theta*(1+time);
        return sin(thetat);
    };

    switch (fp.spaceDimension) {
    case 1: {
        auto nr = static_cast<unsigned int>(fp.N + make_real(0.5));
        cout << "# Generating " << nr << " nodes as a block of size " << nr << "x1x1"
            << ", max.size ratio " << fp.r2/fp.r1
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "F", "xx"});
        wh.writeZoneHeader(nr);
        for (auto ir=0u; ir<nr; ++ir) {
            auto t = lerp(0, 1, ir, nr);
            auto r = fp.r1 + t*(b + t*a);
            auto x = r;
            auto f =
                    // sin(lerp(0, 4*M_PI, ir, nr)) *
                    fieldR(r);
            wh.writeRow(x*fp.sx, f, x);
        }
        break;
    }
    case 2: {
        auto oneElement = std::abs(fp.N - make_real(4)) < make_real(0.1);
        auto nr = oneElement? 2u: static_cast<unsigned int>(pow(xi*fp.N/dphi, make_real(1./2)) + make_real(0.5));
        auto nphi = oneElement? 2u: static_cast<unsigned int>(dphi*nr/xi + 0.5);
        cout << "# Generating " << nr*nphi << " (approx. " << fp.N << ") nodes as a block of size " << nr << "x" << nphi << "x1"
            << ", max.size ratio " << fp.r2/fp.r1
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "Y", "F", "xx"});
        wh.writeZoneHeader(nr, nphi);
        for (auto iphi=0u; iphi<nphi; ++iphi) {
            auto phi = lerp(phi1, phi2, iphi, nphi);
            auto cphi = cos(phi);
            auto sphi = sin(phi);
            for (auto ir=0u; ir<nr; ++ir) {
                auto t = lerp(0, 1, ir, nr);
                auto r = fp.r1 + t*(b + t*a);
                auto x = r*cphi;
                auto y = r*sphi;
                auto f =
                        // sin(lerp(0, 4*M_PI, ir, nr)) *
                        fieldR(r) *
                        fieldPhi(lerp(0, 16*M_PI, iphi, nphi));
                wh.writeRow(x*fp.sx, y*fp.sy, f, x);
            }
        }
        break;
    }
    case 3: {
        auto oneElement = std::abs(fp.N - make_real(8)) < make_real(0.1);
        auto nr = oneElement? 2u: static_cast<unsigned int>(pow(xi*xi*fp.N/(dtheta*dphi), make_real(1./3)) + make_real(0.5));
        auto ntheta = oneElement? 2u: static_cast<unsigned int>(dtheta*nr/xi + 0.5);
        auto nphi = oneElement? 2u: static_cast<unsigned int>(dphi*nr/xi + 0.5);
        cout << "# Generating " << nr*ntheta*nphi << " (approx. " << fp.N << ") nodes as a block of size " << nr << "x" << ntheta << "x" << nphi
            << ", max.size ratio " << fp.r2/fp.r1
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "Y", "Z", "F", "xx"});
        wh.writeZoneHeader(nr, ntheta, nphi);
        for (auto iphi=0u; iphi<nphi; ++iphi) {
            auto phi = lerp(phi1, phi2, iphi, nphi);
            auto cphi = cos(phi);
            auto sphi = sin(phi);
            for (auto itheta=0u; itheta<ntheta; ++itheta) {
                auto theta = lerp(theta1, theta2, itheta, ntheta);
                auto ctheta = cos(theta);
                auto stheta = sin(theta);
                for (auto ir=0u; ir<nr; ++ir) {
                    auto t = lerp(0, 1, ir, nr);
                    auto r = fp.r1 + t*(b + t*a);
                    auto x = r*stheta*cphi;
                    auto y = r*stheta*sphi;
                    auto z = r*ctheta;
                    auto f =
                            // sin(lerp(0, 4*M_PI, ir, nr)) *
                            fieldR(r) *
                            fieldTheta(lerp(0, 8*M_PI, itheta, ntheta)) *
                            fieldPhi(lerp(0, 16*M_PI, iphi, nphi));
                    wh.writeRow(x*fp.sx, y*fp.sy, z*fp.sz, f, x);
                }
            }
        }
        break;
    }
    default:
        throw range_error("Invalid dimension");
    }
}

template <class WriteHelper>
void generateFrameCube(WriteHelper& wh, const FrameParam& fp, real_type time)
{
    auto fieldR = [&time](const real_type& r) {
        auto rt = r*(1+time);
        return static_cast<real_type>(sin(2*M_PI/(5*max(rt, make_real(1e-10)))));
    };
    auto fieldPhi = [&time](const real_type& phi) {
        auto phit = phi*(1+time);
        return sin(phit);
    };
    auto fieldTheta = [&time](const real_type& theta) {
        auto thetat = theta*(1+time);
        return sin(thetat);
    };

    auto R = fp.r2;
    auto tol = make_real(1e-5)*R;
    switch (fp.spaceDimension) {
    case 1: {
        auto nr = static_cast<unsigned int>(fp.N + make_real(0.5));
        cout << "# Generating " << nr << " nodes as a block of size " << nr << "x1x1"
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "F", "xx"});
        wh.writeZoneHeader(nr);
        Refiner rx(fp.fx, 0, make_real(1), nr-1);
        for (auto ir=0u; ir<nr; ++ir) {
            auto r = rx.value();
            wh.writeRow(r*fp.sx, fieldR(r), r);
        }
        break;
    }
    case 2: {
        auto nr = static_cast<unsigned int>(pow(fp.N, make_real(1./2)) + make_real(0.5));
        cout << "# Generating " << nr*nr << " (approx. " << fp.N << ") nodes as a block of size " << nr << "x" << nr << "x1"
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "Y", "F", "lin"});
        wh.writeZoneHeader(nr, nr);
        Refiner ry(fp.fy, -R, 2*R, nr-1);
        for (auto iy=0u; iy<nr; ++iy, ry.next()) {
            auto y = ry.value();
            cout << y << endl;
            Refiner rx(fp.fx, -R, 2*R, nr-1);
            for (auto ix=0u; ix<nr; ++ix, rx.next()) {
                auto x = rx.value();
                auto r = sqrt(x*x + y*y);
                auto phi = fabs(x) + fabs(y) < tol? 0: atan2(y, x);
                auto f = fieldR(r) * fieldPhi(phi*8);
                wh.writeRow(x*fp.sx, y*fp.sy, f, x+2*y);
            }
        }
        break;
    }
    case 3: {
        auto nr = static_cast<unsigned int>(pow(fp.N, make_real(1./3)) + make_real(0.5));
        cout << "# Generating " << nr*nr*nr << " (approx. " << fp.N << ") nodes as a block of size " << nr << "x" << nr << "x" << nr
            << ", time " << time
            << endl;
        wh.writeFileHeader({"X", "Y", "Z", "F", "lin"});
        wh.writeZoneHeader(nr, nr, nr);
        Refiner rz(fp.fz, -R, 2*R, nr-1);
        for (auto iz=0u; iz<nr; ++iz, rz.next()) {
            auto z = rz.value();
            Refiner ry(fp.fy, -R, 2*R, nr-1);
            for (auto iy=0u; iy<nr; ++iy, ry.next()) {
                auto y = ry.value();
                Refiner rx(fp.fx, -R, 2*R, nr-1);
                for (auto ix=0u; ix<nr; ++ix, rx.next()) {
                    auto x = rx.value();
                    auto r = sqrt(x*x + y*y + z*z);
                    auto rxy = sqrt(x*x + y*y);
                    auto phi = fabs(x) + fabs(y) < tol? 0: atan2(y, x);
                    auto theta = fabs(rxy) + fabs(z) < tol? 0: atan2(rxy, z);
                    auto f =
                            fieldR(r) *
                            fieldTheta(theta*8) *
                            fieldPhi(phi*8);
                    wh.writeRow(x*fp.sx, y*fp.sy, z, f, x+2*y+3*z);
                }
            }
        }
        break;
    }
    default:
        throw range_error("Invalid dimension");
    }
}

enum class GeometryType {
    Sphere, Cube
};

template <class WriteHelper>
void generateFrame(GeometryType geomType, WriteHelper& wh, const FrameParam& fp, real_type time)
{
    switch (geomType) {
    case GeometryType::Sphere:
        generateFrameSphere(wh, fp, time);
        break;
    case GeometryType::Cube:
        generateFrameCube(wh, fp, time);
        break;
    }
}

enum class FileType {
    Tecplot,
    Binary
};

} // anonymous namespace

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;
    try {
        FrameParam fp;
        TimeParam tp;
        auto po_value = [](auto& x) {
            return po::value(&x)->default_value(x);
        };
        string outputFileName;
        string geomTypeStr = "sphere";

        po::options_description po_generic("Gerneric options");
        po_generic.add_options()
                ("help,h", "produce help message");

        po::options_description po_frame("Mesh parameters");
        po_frame.add_options()
                ("type", po_value(geomTypeStr), "geometry type (sphere, cube)")
                ("r1", po_value(fp.r1), "inner radius")
                ("r2", po_value(fp.r2), "outer radius")
                ("theta1", po_value(fp.theta1), "starting nutation [deg]")
                ("theta2", po_value(fp.theta2), "ending nutation [deg]")
                ("phi1", po_value(fp.phi1), "starting azimuth [deg]")
                ("phi2", po_value(fp.phi2), "ending azimuth [deg]")
                ("N", po_value(fp.N), "approximate total node count")
                ("sx", po_value(fp.sx), "X scale []")
                ("sy", po_value(fp.sy), "Y scale []")
                ("sz", po_value(fp.sz), "Z scale []")
                ("fx", po_value(fp.fx), "Refinement in X direction []")
                ("fy", po_value(fp.fy), "Refinement in Y direction []")
                ("fz", po_value(fp.fz), "Refinement in Z direction []")
                ("dim", po_value(fp.spaceDimension), "space dimension (1, 2, or 3)");
        po::options_description po_time("Time parameters");
        po_time.add_options()
                ("t1", po_value(tp.startTime), "start time")
                ("t2", po_value(tp.endTime), "end fime")
                ("tn", po_value(tp.frameCount), "frame count");
        po::options_description po_output("Output control");
        po_output.add_options()
                ("output", po::value(&outputFileName), "output file name");
        po::positional_options_description po_pos;
        po_pos.add("output", 1);

        po::variables_map vm;
        auto po_alloptions = po::options_description()
                .add(po_generic).add(po_frame).add(po_time).add(po_output);
        po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions)
                  .positional(po_pos).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << po_alloptions << "\n";
            return 0;
        }

        GeometryType geomType;
        if (geomTypeStr == "sphere")
            geomType = GeometryType::Sphere;
        else if (geomTypeStr == "cube")
            geomType = GeometryType::Cube;
        else
            throw invalid_argument("The 'type' parameter must be either sphere or cube");

        if (tp.frameCount > 1 && !outputFileName.empty())
            experimental::filesystem::create_directory(outputFrameDirectory(outputFileName, true));
        for (auto frame=0u; frame<tp.frameCount; ++frame) {
            cout << "Generating time frame " << (frame+1) << " of " << tp.frameCount << endl;
            ofstream outputFile;
            FileType type = FileType::Tecplot;
            if (!outputFileName.empty()) {
                auto name = frameOutputFileName(outputFileName, frame, tp.frameCount > 1);
                outputFile.open(name, ios::binary);
                if (outputFile.fail())
                    throw runtime_error(string("Failed to open output file '") + name + "'");
                auto ext = experimental::filesystem::path(name).extension();
                if (ext == ".tec")
                    type = FileType::Tecplot;
                else if (ext == ".bin")
                    type = FileType::Binary;
                else
                    throw invalid_argument("Failed to determine file type by name - please specify either .tec or .bin filename extension");
                cout << "Output will be written to file '" << name << "'" << endl;
            }

            auto& os = outputFileName.empty()? cout: outputFile;
            auto time = tp.startTime;
            if (frame > 0)
                time += frame * (tp.endTime - tp.startTime) / (tp.frameCount-1);
            switch (type) {
            case FileType::Tecplot: {
                TecplotMeshWriterHelper wh(os);
                generateFrame(geomType, wh, fp, time);
                break;
            }
            case FileType::Binary: {
                BinaryMeshWriterHelper wh(os, 1);
                generateFrame(geomType, wh, fp, time);
                break;
            }
            }
        }
        return 0;
    }
    catch(const std::exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}
