#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <stdexcept>

#include <boost/noncopyable.hpp>
#include <boost/optional.hpp>

#include "def_prop_class.hpp"
#include "s3dmm/Metadata.hpp"
#include "s3dmm/MeshDataProvider.hpp"
#include "s3dmm/MeshBoundaryExtractor.hpp"
#include "s3dmm/BlockTreeFieldProvider.hpp"
#include "s3dmm/BlockTreeFieldService.hpp"

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithInputFileName, std::string, const std::string&,
        inputFileName, setInputFileName, onInputFileNameChanged, offInputFileNameChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithFieldName, std::string, const std::string&,
        fieldName, setFieldName, onFieldNameChanged, offFieldNameChanged)

template<unsigned int N>
struct BlockTreeLocation
{
    unsigned int level = 0;
    s3dmm::MultiIndex<N, unsigned int> index;
};

template<unsigned int N>
S3DMM_DEF_NOTIFIED_PROP_CLASS(
        WithBlockTreeLocation,
        BlockTreeLocation<N>, const BlockTreeLocation<N>&,
        blockTreeLocation, setBlockTreeLocation,
        onBlockTreeLocationChanged, offBlockTreeLocationChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayQtree, bool, bool,
        displayQtree, setDisplayQtree, onDisplayQtreeChanged, offDisplayQtreeChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithQtreeCompactView, bool, bool,
        qtreeCompactView, setQtreeCompactView, onQtreeCompactViewChanged, offQtreeCompactViewChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithQtreeDisplayIds, bool, bool,
        qtreeDisplayIds, setQtreeDisplayIds, onQtreeDisplayIdsChanged, offQtreeDisplayIdsChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithLimitedQtree, bool, bool,
        limitedQtree, setLimitedQtree, onLimitedQtreeChanged, offLimitedQtreeChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayMesh, bool, bool,
        displayMesh, setDisplayMesh, onDisplayMeshChanged, offDisplayMeshChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayBoundary, bool, bool,
        displayBoundary, setDisplayBoundary, onDisplayBoundaryChanged, offDisplayBoundaryChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayBoundaryNodeNumbers, bool, bool,
        displayBoundaryNodeNumbers, setDisplayBoundaryNodeNumbers, onDisplayBoundaryNodeNumbersChanged, offDisplayBoundaryNodeNumbersChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayBoundaryDirMarkers, bool, bool,
        displayBoundaryDirMarkers, setDisplayBoundaryDirMarkers, onDisplayBoundaryDirMarkersChanged, offDisplayBoundaryDirMarkersChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplaySparseNodes, bool, bool,
        displaySparseNodes, setDisplaySparseNodes, onDisplaySparseNodesChanged, offDisplaySparseNodesChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplaySparseNodeNumbers, bool, bool,
        displaySparseNodeNumbers, setDisplaySparseNodeNumbers, onDisplaySparseNodeNumbersChanged, offDisplaySparseNodeNumbersChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithDisplayDenseField, bool, bool,
        displayDenseField, setDisplayDenseField, onDisplayDenseFieldChanged, offDisplayDenseFieldChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithFill, bool, bool,
        fill, setFill, onFillChanged, offFillChanged)

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithQuadtreeNodeFill, bool, bool,
        quadtreeNodefill, setQuadtreeNodeFill, onQuadtreeNodeFillChanged, offQuadtreeNodeFillChanged)

template<unsigned int N>
class StateDataTemplate :
    boost::noncopyable,
    public WithInputFileName,
    public WithFieldName,
    public WithBlockTreeLocation<N>,
    public WithDisplayQtree,
    public WithQtreeCompactView,
    public WithQtreeDisplayIds,
    public WithLimitedQtree,
    public WithDisplayMesh,
    public WithDisplayBoundary,
    public WithDisplayBoundaryNodeNumbers,
    public WithDisplayBoundaryDirMarkers,
    public WithDisplaySparseNodes,
    public WithDisplaySparseNodeNumbers,
    public WithDisplayDenseField,
    public WithFill,
    public WithQuadtreeNodeFill
{
public:
    using Metadata = s3dmm::Metadata<N>;
    using MeshDataProvider = s3dmm::MeshDataProvider;
    using MeshBoundaryExtractor = s3dmm::MeshBoundaryExtractor;
    using BlockTreeNodes = s3dmm::BlockTreeNodes<N, typename Metadata::BT>;
    using BlockTreeFieldProvider = s3dmm::BlockTreeFieldProvider<N>;
    using BlockTreeFieldService = s3dmm::BlockTreeFieldService<N>;

    StateDataTemplate() :
        WithDisplayQtree(true),
        WithQtreeCompactView(true),
        WithQtreeDisplayIds(true),
        WithLimitedQtree(false),
        WithDisplayMesh(false),
        WithDisplayBoundary(true),
        WithDisplayBoundaryNodeNumbers(false),
        WithDisplayBoundaryDirMarkers(false),
        WithDisplaySparseNodes(true),
        WithDisplaySparseNodeNumbers(true),
        WithDisplayDenseField(false),
        WithFill(true),
        WithQuadtreeNodeFill(true)
    {
        onInputFileNameChanged([this] { reset(); });
        onFieldNameChanged([this] { resetField(); });
        this->onBlockTreeLocationChanged([this] { resetRootSubtree(); });
    }

    const Metadata& metadata() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("s3dmm metadata is not available because the input file name is not set");
        if (!m_metadata) {
            auto is = openInputStream(inputFileName() + ".s3dmm-meta");
            m_metadata = std::make_unique<Metadata>(is);
        }
        return *m_metadata;
    }

    MeshDataProvider& mesh() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Mesh is not available because the input file name is not set");
        if (!m_mesh)
            m_mesh = std::make_unique<MeshDataProvider>(inputFileName());
        return *m_mesh;
    }

    MeshBoundaryExtractor& boundary() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Mesh boundary is not available because the input file name is not set");
        if (!m_boundary)
            m_boundary = std::make_unique<MeshBoundaryExtractor>(mesh(), s3dmm::MeshElementRefinerParam());
        return *m_boundary;
    }

    const BlockTreeNodes& rootSubtreeNodes() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Root subtree nodes are not available because the input file name is not set");
        if (!m_rootSubtreeNodes) {
            auto& md = metadata();
            auto& bt = md.blockTree();
            auto rootLocation = this->blockTreeLocation();
            auto root = bt.blockAt(rootLocation.index, rootLocation.level);
            m_rootSubtreeNodes = std::make_unique<BlockTreeNodes>(metadata().blockTreeNodes(root));
        }
        return *m_rootSubtreeNodes;
    }

    bool hasSparseField() const {
        return !inputFileName().empty() && !fieldName().empty();
    }

    const std::vector<s3dmm::real_type>& sparseField() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Sparse field is not available because the input file name is not set");
        if (fieldName().empty())
            throw std::runtime_error("Sparse field is not available because field name is not set");
        if (!m_fieldProvider) {
            auto fieldFileName = inputFileName() + ".s3dmm-field#" + fieldName();
            m_fieldProvider = std::make_unique<BlockTreeFieldProvider>(metadata(), fieldFileName);
        }
        s3dmm::Vec2<s3dmm::real_type> fieldRange;
        m_fieldProvider->fieldValues(fieldRange, m_sparseField, rootSubtreeNodes());
        if (fieldRange[1] >= fieldRange[0])
            m_fieldRange = fieldRange.convertTo<s3dmm::dfield_real>();
        else
            m_fieldRange = s3dmm::Vec<2, s3dmm::dfield_real>{ 0, 0 };
        return m_sparseField;
    }

    s3dmm::Vec<2, s3dmm::dfield_real> fieldRange() const
    {
        if (!m_fieldRange) {
            sparseField();
            BOOST_ASSERT(!!m_fieldRange);
        }
        return m_fieldRange.get();
    }

    BlockTreeFieldService& fieldService() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Field service is not available because the input file name is not set");
        if (!m_fieldService)
            m_fieldService = std::make_unique<BlockTreeFieldService>(inputFileName());
        return *m_fieldService;
    }

    const std::vector<s3dmm::dfield_real>& denseField() const
    {
        if (inputFileName().empty())
            throw std::runtime_error("Dense field is not available because the input file name is not set");
        if (fieldName().empty())
            throw std::runtime_error("Dense field is not available because field name is not set");
        if (m_denseField.empty())
            s3dmm::DenseFieldInterpolator<N>::interpolate(m_denseField, sparseField(), rootSubtreeNodes());
        return m_denseField;
    }

    unsigned int rootSubtreeDepth() const
    {
        auto rootLocation = this->blockTreeLocation();
        return m_metadata->subtreeDepth(rootLocation.level, rootLocation.index);
    }

private:
    mutable std::unique_ptr<Metadata> m_metadata;
    mutable std::unique_ptr<MeshDataProvider> m_mesh;
    mutable std::unique_ptr<MeshBoundaryExtractor> m_boundary;
    mutable std::unique_ptr<BlockTreeNodes> m_rootSubtreeNodes;
    mutable std::unique_ptr<BlockTreeFieldProvider> m_fieldProvider;
    mutable std::unique_ptr<BlockTreeFieldService> m_fieldService;

    mutable std::vector<s3dmm::real_type> m_sparseField;
    mutable boost::optional<s3dmm::Vec<2, s3dmm::dfield_real>> m_fieldRange;
    mutable std::vector<s3dmm::dfield_real> m_denseField;

    void reset() {
        m_metadata.reset();
        m_mesh.reset();
        m_boundary.reset();
        m_fieldService.reset();
        resetRootSubtree();
    }

    void resetRootSubtree()
    {
        m_rootSubtreeNodes.reset();
        resetField();
    }

    void resetField()
    {
        m_fieldProvider.reset();
        m_fieldRange.reset();
        m_denseField.clear();
    }

    static std::ifstream openInputStream(const std::string& fileName)
    {
        std::ifstream is(fileName);
        if (!is.is_open())
            throw std::runtime_error(std::string("Failed to open input file '") + fileName + "'");
        is.exceptions(std::ios::failbit);
        return is;
    }
};

using StateData = StateDataTemplate<2>;
