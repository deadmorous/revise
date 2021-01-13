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

#include "vtk_rect_grid.hpp"

#include "s3dmm/RectGridData.hpp"
#include "s3dmm/ProgressReport.hpp"

#include "silver_bullets/templatize/resolve_template_args.hpp"

#include <vtkSmartPointer.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h>
#include <vtkAOSDataArrayTemplate.h>
#include <vtkCellData.h>
#include <vtkDataArrayRange.h>

#include <boost/assert.hpp>

using namespace std;
using namespace s3dmm;

namespace {

template<int VtkDataType> struct VtkDataTypeMap;
template<int VtkDataType> using VtkDataTypeMap_t = typename VtkDataTypeMap<VtkDataType>::type;

#define DECL_SUPPORTED_VTK_DATA_TYPE(VtkDataType, actualType) \
    template<> struct VtkDataTypeMap<VtkDataType> { using type = actualType; }
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_CHAR, char);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_SIGNED_CHAR, char);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_UNSIGNED_CHAR, unsigned char);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_SHORT, short);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_UNSIGNED_SHORT, unsigned short);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_INT, int);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_UNSIGNED_INT, unsigned int);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_LONG, long);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_UNSIGNED_LONG, unsigned long);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_FLOAT, float);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_DOUBLE, double);
DECL_SUPPORTED_VTK_DATA_TYPE(VTK_ID_TYPE, vtkIdType);

using SupportedVtkDataTypes = integer_sequence<int,
    VTK_CHAR,
    VTK_SIGNED_CHAR,
    VTK_UNSIGNED_CHAR,
    VTK_SHORT,
    VTK_UNSIGNED_SHORT,
    VTK_INT,
    VTK_UNSIGNED_INT,
    VTK_LONG,
    VTK_UNSIGNED_LONG,
    VTK_FLOAT,
    VTK_DOUBLE,
    VTK_ID_TYPE>;

template<unsigned int N, unsigned int VtkTypeId>
inline void addArrayInfoTemplate(RectGridData<N>& rgd, const string& name, std::size_t size) {
    rgd.template addArrayInfo<VtkDataTypeMap_t<VtkTypeId>>(name, size);
}

template<unsigned int N>
struct callAddArrayInfoTemplate {
    template<unsigned int VtkTypeId> void operator()(RectGridData<N>& rgd, const string& name, std::size_t size) const {
        addArrayInfoTemplate<N, VtkTypeId>(rgd, name, size);
    }
};

template<unsigned int N, unsigned int VtkTypeId>
inline void writeArrayTemplate(RectGridData<N>& rgd, vtkAbstractArray *arr)
{
    using T = VtkDataTypeMap_t<VtkTypeId>;
    using A = vtkAOSDataArrayTemplate<T>;
    auto a = A::SafeDownCast(arr);
    BOOST_ASSERT(a);
    rgd.template writeArray<T>(vtk::DataArrayValueRange(a));
}

template<unsigned int N>
struct callWriteArrayTemplate {
    template<unsigned int VtkTypeId> void operator()(RectGridData<N>& rgd, vtkAbstractArray *arr) const {
        writeArrayTemplate<N, VtkTypeId>(rgd, arr);
    }
};

template <unsigned int N>
void convertVtkRectGridTemplate(const RunParameters& param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Read VTK file");
    auto reader = vtkSmartPointer<vtkRectilinearGridReader>::New();
    reader->SetFileName(param.inputFileName.c_str());
    reader->Update();

    REPORT_PROGRESS_STAGE("Process VTK file");
    auto out = reader->GetOutput();
    auto dataDimension = out->GetDataDimension();
    if (dataDimension != N)
        throw runtime_error(
            "Wrong data dimension: expected " +
            to_string(N) + ", found " + to_string(dataDimension));

    auto *gridSizePtr = out->GetDimensions();
    MultiIndex<N, unsigned int> gridSize;
    copy(gridSizePtr, gridSizePtr+N, gridSize.data());

    RectGridData<N> rgd;
    rgd.setGridSize(gridSize);

    MultiIndex<N, std::vector<real_type>> gridCoordinates;
    auto getAxisCoordinate = [&gridCoordinates](unsigned int d, vtkDataArray *c) {
        auto n = c->GetNumberOfValues();
        auto& dst = gridCoordinates.at(d);
        dst.resize(n);
        for (auto i=0; i<n; ++i)
            dst[i] = c->GetTuple1(i);
    };
    if (N > 0)
        getAxisCoordinate(0, out->GetXCoordinates());
    if (N > 1)
        getAxisCoordinate(0, out->GetYCoordinates());
    if (N > 2)
        getAxisCoordinate(0, out->GetZCoordinates());
    BOOST_STATIC_ASSERT(N <= 3);
    rgd.setGridCoordinates(gridCoordinates);

    auto arrayCount = out->GetCellData()->GetNumberOfArrays();
    auto cd = out->GetCellData();
    for (auto iarr=0; iarr<arrayCount; ++iarr) {
        auto name = cd->GetArrayName(iarr);
        auto arr = cd->GetAbstractArray(iarr);
        auto size = arr->GetNumberOfValues();
        silver_bullets::resolve_template_args<SupportedVtkDataTypes>(
            make_tuple(arr->GetDataType()), callAddArrayInfoTemplate<N>(), rgd, name, size);
    }

    REPORT_PROGRESS_STAGE("Write output binary file");
    rgd.openForWrite(param.outputFileName);
    for (auto iarr=0; iarr<arrayCount; ++iarr) {
        auto arr = cd->GetAbstractArray(iarr);
        silver_bullets::resolve_template_args<SupportedVtkDataTypes>(
            make_tuple(arr->GetDataType()), callWriteArrayTemplate<N>(), rgd, arr);
    }
}

struct callConvertVtkRectGridTemplate {
    template<unsigned int N> void operator()(const RunParameters& param) const {
        convertVtkRectGridTemplate<N>(param);
    }
};

} // anonymous namespace

void convertVtkRectGrid(const RunParameters &param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Run the entire job");
    silver_bullets::resolve_template_args<
        integer_sequence<unsigned int, 1,2,3>>(
        make_tuple(param.spaceDimension), callConvertVtkRectGridTemplate(), param);
}
