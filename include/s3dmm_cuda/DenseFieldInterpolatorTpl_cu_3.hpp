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

#pragma once

#include "defs.h"
#include "s3dmm/BlockTreeFieldProvider.hpp"
#include "s3dmm/IndexTransform.hpp"
#include "s3dmm/HypercubeWalker.hpp"
#include "s3dmm/IncMultiIndex.hpp"
#include "s3dmm/DenseFieldInterpolatorTimers.hpp"

#include "s3dmm/FieldTimestamps.hpp"

// For some unknown reason, uncommenting this macro has the following effect (on Ubuntu 18.04):
// - when sparse field is read from SSD, the entire process speeds up several percent;
// - when sparse field is read from HDD, the entire process slows down several percent;
// When the macro is commented out, HDD outerpforms SSD, which looks like "latency hiding" applied
// to the reading of the sparse field.
// #define S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ

namespace s3dmm {

template<class ContainerRef>
class DenseFieldInterpolatorTpl_cu_3
{
public:
    static constexpr const unsigned int N = 3;
    using BlockId = typename BlockTree<N>::BlockId;
    using BlockIndex = typename BlockTree<N>::BlockIndex;
    using BT = typename Metadata<N>::BT;
    using dense_field_container_ref = ContainerRef;
    using Timers = DenseFieldInterpolatorTimers;

    std::shared_ptr<Timers> timers() const {
        return m_timers;
    }
    void setTimers(const std::shared_ptr<Timers>& timers) {
        m_timers = timers;
    }

    DenseFieldInterpolatorTpl_cu_3(BlockTreeFieldProvider<N>& fp) : m_fp(fp)
    {}

    void interpolate(
            Vec2<real_type>& fieldRange,
            dense_field_container_ref denseFieldDev,
            const BlockId& subtreeRoot)
    {
        // Obtain subtree and its nodes.
        ScopedTimerUser blockTreeNodesTimerUser(m_timers? &m_timers.get()->blockTreeNodesTimer: nullptr);
        auto& md = m_fp.metadata();
        auto subtreeNodes = md.blockTreeNodes(subtreeRoot); // TODO better: Copying!!!
        blockTreeNodesTimerUser.stop();

        interpolate(fieldRange, denseFieldDev, subtreeNodes);
    }

    void interpolate(
            Vec2<real_type>& fieldRange,
            dense_field_container_ref denseFieldDev,
            const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        // Obtain sparseField
        ScopedTimerUser sparseFieldTimerUser(m_timers? &m_timers.get()->sparseFieldTimer: nullptr);
        auto& sparseField = m_buf;
        m_fp.fieldValues(fieldRange, sparseField, subtreeNodes);
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ
        if (!sparseField.empty()) {
            volatile real_type x;
            x = sparseField.front();
            x = sparseField.back();
        }
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ
        sparseFieldTimerUser.stop();
        m_timestamps.afterReadSparseField = hires_time();

        if (m_timers)
            ++m_timers.get()->invocationCount;

        // Interpolate dense field
        ScopedTimerUser interpolateTimerUser(m_timers? &m_timers.get()->interpolateTimer: nullptr);
        interpolate(denseFieldDev, sparseField, subtreeNodes);
        m_timestamps.afterComputeDenseField = hires_time();
    }

    static S3DMM_CUDA_API void interpolate(
            dense_field_container_ref denseFieldDev,
            const std::vector<real_type>& sparseField,
            const BlockTreeNodes<N, BT>& subtreeNodes
        );

    const FieldTimestamps& timestamps() const {
        return m_timestamps;
    }

private:
    BlockTreeFieldProvider<N>& m_fp;
    std::vector<real_type> m_buf;
    std::shared_ptr<Timers> m_timers;
    FieldTimestamps m_timestamps;
};

} // s3dmm
