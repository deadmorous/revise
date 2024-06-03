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

#include "Imamod_gMeshReader.h"

#include "binary_io.hpp"

#include <boost/assert.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>

#include <filesystem>
#include <map>
#include <regex>
#include <set>

using namespace std;

namespace s3dmm {

namespace {

template<class ... Args>
ifstream openInputFile(const string& fileName, Args ... args)
{
    ifstream result(fileName, args...);
    if (!result.is_open())
        throw runtime_error(string("Failed to open input file '") + fileName + "'");
    return result;
}

map<string, string> readValuesFromTextStream(istream& s, const set<string>& valueNames)
{
    map<string, string> result;
    regex rx("^\\s*(\\w+)\\s+(\\S.*)\\s*$");
    while (!s.eof()) {
        string line;
        getline(s, line);
        smatch m;
        if (regex_match(line, m, rx)) {
            string name = m[1];
            if (valueNames.find(name) != valueNames.end()) {
                if (result.find(name) != result.end())
                    throw runtime_error("Value '" + name + " is specified more than once");
                result[name] = m[2];
            }
        }
    }
    if (result.size() < valueNames.size()) {
        vector<string> missingValues;
        for (auto name : valueNames) {
            if (result.find(name) == result.end())
                missingValues.push_back(name);
        }
        BOOST_ASSERT(!missingValues.empty());
        ostringstream oss;
        oss << "The following values are not specified: " << boost::join(missingValues, ", ");
        throw runtime_error(oss.str());
    }
    return result;
}

} // anonymous namespace

tecplot_rw::TecplotHeader Imamod_gMeshReader::GetHeader() const
{
    tecplot_rw::TecplotHeader result;
    result.type = tecplot_rw::TECPLOT_FT_FULL;
    result.title = m_zoneInfo.at(0).title;
    result.vars = m_vars;
    return result;
}

const tecplot_rw::ZoneInfo_v &Imamod_gMeshReader::GetZoneInfo() const
{
    return m_zoneInfo;
}

void Imamod_gMeshReader::GetZoneVariableValues(size_t nZone, double *pbuf) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

void Imamod_gMeshReader::GetZoneVariableValues(size_t nZone, float *pbuf) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

void Imamod_gMeshReader::GetZoneConnectivity(size_t nZone, unsigned int *pbuf) const
{
    BOOST_ASSERT(nZone == 0);
    boost::ignore_unused_variable_warning(nZone);

    auto fe = openInputFile(m_elementNodeFileName, ios::binary);

    // Skip 12 byte header in the node numbers file
    fe.seekg(12, ios::beg);

    // Read element node numbers
    auto& zoneInfo = m_zoneInfo.at(0);
    auto p = pbuf;
    for (size_t ie=0; ie<zoneInfo.uNumElem; ++ie, p+=8) {
        fe.read(reinterpret_cast<char*>(p), 8*sizeof(unsigned int));
        if (p[7] != ~0u) {
            swap(p[2], p[3]);
            swap(p[6], p[7]);
        }
    }

    // deBUG, TODO: Remove
//    writeElementSamples(pbuf);
//    writeMinifiedProblem(pbuf);
}

void Imamod_gMeshReader::Open(const string &file)
{
    Close();
    m_ownStream = make_shared<ifstream>(file, ios::binary);
    if (!m_ownStream->is_open())
        throw runtime_error(string("Failed to open input file '") + file + "'");
    Attach(*m_ownStream);
}

void Imamod_gMeshReader::Attach(istream &is)
{
    // Read meta-information file
    is.exceptions(istream::failbit);
    m_s = &is;
    auto readLine = [this]() {
        string result;
        getline(*m_s, result);
        return result;
    };
    auto meshInfoFileName = readLine();
    m_nodeCoordFileName = readLine();
    m_elementNodeFileName = readLine();
    auto fieldInfoFileName = readLine();
    m_fieldFileName = readLine();

    m_zoneInfo.resize(1);
    auto& zoneInfo = m_zoneInfo[0] = tecplot_rw::ZoneInfo();
    zoneInfo.bIsOrdered = false;
    zoneInfo.bPoint = false;

    // Read mesh info file
    {
        auto meshInfoFile = openInputFile(meshInfoFileName);
        auto meshInfoValues = readValuesFromTextStream(
            meshInfoFile, {"NumCoords", "NnBase", "NtBase", "Name"});
        m_dim = boost::lexical_cast<unsigned int>(meshInfoValues.at("NumCoords"));
        if (m_dim > 0)
            m_vars.push_back("x");
        if (m_dim > 1)
            m_vars.push_back("y");
        if (m_dim > 2)
            m_vars.push_back("z");
        if (m_dim < 2   ||   m_dim > 3)
            throw runtime_error(
                "Invalid dimension is specified in file '" +
                meshInfoFileName +
                "' (only 3 is currently supported)");
        zoneInfo.title = meshInfoValues.at("Name");
        zoneInfo.uNumNode = boost::lexical_cast<size_t>(meshInfoValues.at("NnBase"));
        zoneInfo.uNumElem = boost::lexical_cast<size_t>(meshInfoValues.at("NtBase"));
        zoneInfo.uElemType = tecplot_rw::TECPLOT_ET_BRICK;
    }

    // Read field info file
    {
        auto fieldInfoFile = openInputFile(fieldInfoFileName);
        while (true) {
            string word;
            fieldInfoFile >> word;
            if (fieldInfoFile.fail())
                throw runtime_error("Failed to parse field info flie '" + fieldInfoFileName + "'");
            if (word == "Variables")
                break;
        }
        while (true) {
            string word;
            fieldInfoFile >> word;
            if (fieldInfoFile.fail())
                break;
            if (word.size() > 1 && word[0] == '"' && word[word.size()-1] == '"')
                word = word.substr(1, word.size()-2);
            m_vars.push_back(word);
        }
    }

    zoneInfo.uBufSizeVar = m_vars.size() * zoneInfo.uNumNode;
    zoneInfo.uBufSizeCnt = zoneInfo.uNumElem * 8;
}

void Imamod_gMeshReader::Close()
{
    m_s = nullptr;
    m_ownStream.reset();
    m_zoneInfo.clear();
    m_vars.clear();
    m_nodeCoordFileName.clear();
    m_elementNodeFileName.clear();
    m_fieldFileName.clear();
}

template<class T>
void Imamod_gMeshReader::GetZoneVariableValuesPriv(size_t nZone, T *pbuf) const
{
    BOOST_ASSERT(nZone == 0);
    boost::ignore_unused_variable_warning(nZone);

    auto fc = openInputFile(m_nodeCoordFileName, ios::binary);
    auto fv = openInputFile(m_fieldFileName, ios::binary);

    // Skip 12 byte header in the coordinates file
    fc.seekg(12, ios::beg);
    BinaryReader rc(fc);
    BinaryReader rv(fv);

    auto& zoneInfo = m_zoneInfo.at(0);
    BOOST_ASSERT(m_vars.size() > m_dim);
    auto fieldVarCount = m_vars.size() - m_dim;
    auto p = pbuf;
    for (size_t inode=0; inode < zoneInfo.uNumNode; ++inode) {
        for (auto d=0u; d<m_dim; ++d, ++p)
            *p = static_cast<T>(rc.read<double>());
        for (size_t iv=0; iv<fieldVarCount; ++iv, ++p)
            *p = static_cast<T>(rv.read<float>());
    }

    // deBUG, TODO: Remove
    writeFieldSamples(pbuf);
}

// deBUG, TODO: Remove
template<class T>
void Imamod_gMeshReader::writeFieldSamples(const T *pbuf) const
{
    static bool samplesWritten = false;
    if (samplesWritten)
        return;
    samplesWritten = true;

    ofstream fs("field_samples.txt");
    auto& zoneInfo = m_zoneInfo.at(0);
    auto p = pbuf;
    for (size_t ivar=0; ivar<m_vars.size(); ++ivar) {
        if (ivar > 0)
            fs << '\t';
        fs << m_vars[ivar];
    }
    fs << endl;

    auto n = min(zoneInfo.uNumNode, size_t(10));
    for (size_t iv=0; iv<n; ++iv, p+=m_vars.size()) {
        for (size_t ivar=0; ivar<m_vars.size(); ++ivar) {
            if (ivar > 0)
                fs << '\t';
            fs << p[ivar];
        }
        fs << endl;
    }
}

// deBUG, TODO: Remove
void Imamod_gMeshReader::writeElementSamples(const unsigned int *pbuf) const
{
    static bool samplesWritten = false;
    if (samplesWritten)
        return;
    samplesWritten = true;

    auto fc = openInputFile(m_nodeCoordFileName, ios::binary);

    // Skip 12 byte header in the coordinates file
    fc.seekg(12, ios::beg);
    BinaryReader rc(fc);

    auto& zoneInfo = m_zoneInfo.at(0);
    std::array<size_t, 8> npadFound;
    npadFound.fill(0);
    ofstream fs("element_samples.txt");
    auto p = pbuf;
    for (size_t ie=0; ie<zoneInfo.uNumElem; ++ie, p+=8) {
        auto npad = count_if(p, p+8, [](unsigned int iv) { return iv == ~0u; });
        ++npadFound[npad];
        if (npadFound[npad] > 1)
            continue;
        for (auto i=0; i<8; ++i) {
            auto iv = p[i];
            if (iv == ~0u)
                fs << "-" << endl;
            else {
                fc.seekg(12 + iv*m_dim*sizeof(double));
                for (auto ic=0u; ic<m_dim; ++ic) {
                    if (ic > 0)
                        fs << '\t';
                    fs << rc.read<double>();
                }
                fs << endl;
            }
        }
        fs << endl;
    }

    for (auto i=0; i<8; ++i) {
        if (npadFound[i] > 0) {
            fs << "Elements with " << (8-i) << " nodes: " << npadFound[i] << endl;
        }
    }

    auto ivmin = min_element(pbuf, pbuf + zoneInfo.uNumElem*8);
    fs << endl << "First node has number " << ivmin << endl;
}

void Imamod_gMeshReader::writeMinifiedProblem(const unsigned int *pbuf) const
{
    static bool minifiedProblemWritten = false;
    if (minifiedProblemWritten)
        return;
    minifiedProblemWritten = true;

    auto& zoneInfo = m_zoneInfo.at(0);
    constexpr size_t SampleProblemElementCount = 1000;
    if (zoneInfo.uNumElem <= SampleProblemElementCount)
        return;

    // Generate new node numbers
    std::map<unsigned int, unsigned int> nodeOld2New;
    auto p = pbuf;
    auto newNodeCount = 0u;
    for (size_t ie=0; ie<SampleProblemElementCount; ++ie, p+=8) {
        for (auto i=0; i<8; ++i) {
            auto iv = p[i];
            if (iv != ~0u && nodeOld2New.find(iv) == nodeOld2New.end())
                nodeOld2New[iv] = newNodeCount++;
        }
    }
    std::map<unsigned int, unsigned int> nodeNew2Old;
    for (auto& item : nodeOld2New)
        nodeNew2Old[item.second] = item.first;
    BOOST_ASSERT(nodeNew2Old.size() == nodeOld2New.size());

    auto problemDirName = "sample_" + boost::lexical_cast<string>(SampleProblemElementCount);
    filesystem::create_directory(problemDirName);

    // Write element node numbers
    {
        ofstream ofe(problemDirName + "/min_topo.msb", ios::binary);
        // Write 12 byte header
        ofe.write("012345678901", 12);
        p = pbuf;
        for (size_t ie=0; ie<SampleProblemElementCount; ++ie, p+=8) {
            unsigned int ev[8];
            for (auto i=0; i<8; ++i) {
                auto iv = p[i];
                if (iv != ~0u)
                    iv = nodeOld2New.at(iv);
                ev[i] = iv;
            }
            if (ev[7] != ~0u) {
                swap(ev[2], ev[3]);
                swap(ev[6], ev[7]);
            }
            ofe.write(reinterpret_cast<const char*>(ev), 8*sizeof (unsigned int));
        }
    }

    // Write node coordinates
    {
        auto fc = openInputFile(m_nodeCoordFileName, ios::binary);
        BinaryReader rc(fc);
        ofstream ofc(problemDirName + "/min_coordinate.msb", ios::binary);
        // Write 12 byte header
        ofc.write("012345678901", 12);
        BinaryWriter wc(ofc);
        for (auto& n2n : nodeNew2Old) {
            auto iv = n2n.second;
            fc.seekg(12 + iv*m_dim*sizeof(double));
            for (auto ic=0u; ic<m_dim; ++ic)
                wc << rc.read<double>();
        }
    }

    // Write fields
    {
        auto fv = openInputFile(m_fieldFileName, ios::binary);
        BinaryReader rv(fv);
        ofstream ofv(problemDirName + "/min_tec_200k_data.dat", ios::binary);
        BOOST_ASSERT(m_vars.size() > m_dim);
        auto fieldVarCount = m_vars.size() - m_dim;
        BinaryWriter wv(ofv);
        for (auto& n2n : nodeNew2Old) {
            auto iv = n2n.second;
            fv.seekg(12 + iv*fieldVarCount*sizeof(float));
            for (auto ivar=0u; ivar<fieldVarCount; ++ivar)
                wv << rv.read<float>();
        }
    }

    // Write meta-information files
    {
        ofstream ofmm(problemDirName + "/min_mesh.txt");
        ofmm
            << "Name" << " " << zoneInfo.title << endl
            << "NumCoords" << " " << m_dim << endl
            << "NnBase" << " " << nodeNew2Old.size() << endl
            << "NtBase" << " " << SampleProblemElementCount << endl;
        ofstream ofvm(problemDirName + "/min_tec_200k_data.dat.a");
        ofvm << "Variables" << endl
             << boost::join( boost::make_iterator_range(m_vars.begin() + m_dim, m_vars.end()), " ") << endl;
        ofstream ofmain(problemDirName + "/example.imm-g");
        ofmain << "min_mesh.txt" << endl
               << "min_coordinate.msb" << endl
               << "min_topo.msb" << endl
               << "min_tec_200k_data.dat.a" << endl
               << "min_tec_200k_data.dat" << endl;
    }
}

}
