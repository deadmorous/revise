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

#include "VolumeTextureDataInterface.hpp"

#ifdef S3VS_WORKER_REPORT_DFIELD_TIME
#define DFIELD_REPORT_PROGRESS_STAGES REPORT_PROGRESS_STAGES
#define DFIELD_REPORT_PROGRESS_STAGE REPORT_PROGRESS_STAGE
#define DFIELD_REPORT_PROGRESS_END REPORT_PROGRESS_END
#else // S3VS_WORKER_REPORT_DFIELD_TIME
#define DFIELD_REPORT_PROGRESS_STAGES(...)
#define DFIELD_REPORT_PROGRESS_STAGE(...)
#define DFIELD_REPORT_PROGRESS_END(...)
#endif

class VolumeTextureDataBase : public VolumeTextureDataInterface
{
public:
    VolumeTextureDataBase() {
        setUpdated(false);
    }

    void setFieldService(s3dmm::BlockTreeFieldService<3> *fieldService) override
    {
        m_fieldService = fieldService;
        if (fieldService)
            m_depth = m_fieldService->metadata().subtreeDepth(BlockId());
        else
            m_depth = DefaultFieldDepth;
        setUpdated(false);
    }

    void setField(
            unsigned int fieldIndex, unsigned int timeFrame,
            const BlockId& subtreeRoot) override
    {
        m_fieldId = {fieldIndex, timeFrame, subtreeRoot};
        m_depth = m_fieldService->metadata().subtreeDepth(subtreeRoot);
    }

    unsigned int depth() const override {
        return m_depth;
    }

    s3dmm::BlockTreeFieldService<3> *fieldService() const {
        return m_fieldService;
    }

protected:
    bool needUpdate() const {
        return m_computedFieldId != m_fieldId;
    }

    void setUpdated(bool updated) const {
        m_computedFieldId = updated? m_fieldId: FieldId{ ~0u, ~0u, BlockId() };
    }

    struct FieldId
    {
        unsigned int fieldIndex = 0u;
        unsigned int timeFrame = 0u;
        BlockId subtreeRoot;
        bool operator==(const FieldId& that) const
        {
            return  fieldIndex == that.fieldIndex &&
                    timeFrame == that.timeFrame &&
                    subtreeRoot == that.subtreeRoot;
        }
        bool operator!=(const FieldId& that) const {
            return !(*this == that);
        }
    };

    const FieldId& fieldId() const {
        return m_fieldId;
    }

    static unsigned int computeDefaultField(std::vector<s3dmm::dfield_real>& field)
    {
        auto N = 1 << DefaultFieldDepth;
        auto L = 10.f;
        field.resize((N+1)*(N+1)*(N+1));
        auto pixels = field.data();
        for (auto i = 0u; i <= N; ++i)
        {
            auto x = s3dmm::make_dfield_real(i) / N;
            for (auto j = 0u; j <= N; ++j)
            {
                auto y = s3dmm::make_dfield_real(j) / N;
                for (auto k = 0u; k <= N; ++k)
                {
                    auto z = s3dmm::make_dfield_real(k) / N;
                    auto v = 0.5f
                             * (sin(L * (x - 0.5f)) * sin(L * (y - 0.5f))
                                    * sin(L * (z - 0.5f))
                                + 1);
                    *pixels++ = v;
                }
            }
        }
        return 8;
    }

    static constexpr const unsigned int DefaultFieldDepth = 5;

private:
    FieldId m_fieldId;

    mutable FieldId m_computedFieldId;

    s3dmm::BlockTreeFieldService<3> *m_fieldService = nullptr;
    mutable unsigned int m_depth = DefaultFieldDepth;
};
