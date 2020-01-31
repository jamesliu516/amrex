
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFabUtil_C.H>
#include <AMReX_EBMultiFabUtil_C.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_MultiCutFab.H>

#include <AMReX_VisMF.H>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace amrex
{

void
EB_set_covered (MultiFab& mf, Real val)
{
    EB_set_covered(mf, 0, mf.nComp(), 0, val);
}

void
EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow, Real val)
{
    const auto factory = dynamic_cast<EBFArrayBoxFactory const*>(&(mf.Factory()));
    if (factory == nullptr) return;
    const auto& flags = factory->getMultiEBCellFlagFab();

    AMREX_ALWAYS_ASSERT(mf.ixType().cellCentered() || mf.ixType().nodeCentered());
    bool is_cell_centered = mf.ixType().cellCentered();
    int ng = std::min(mf.nGrow(),ngrow);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);
        const auto& flagarr = flags.const_array(mfi);
        Array4<Real> const& arr = mf.array(mfi);

        if (is_cell_centered) {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                if (flagarr(i,j,k).isCovered()) {
                    arr(i,j,k,n+icomp) = val;
                }
            });
        } else {
            AMREX_HOST_DEVICE_FOR_4D (bx, ncomp, i, j, k, n,
            {
                eb_set_covered_nodes(i,j,k,n,icomp,arr,flagarr,val);
            });
        }
    }
}

void
EB_set_covered (MultiFab& mf, int icomp, int ncomp, const Vector<Real>& vals)
{
    EB_set_covered(mf, icomp, ncomp, 0, vals);
}

void
EB_set_covered (MultiFab& mf, int icomp, int ncomp, int ngrow, const Vector<Real>& a_vals)
{
    const auto factory = dynamic_cast<EBFArrayBoxFactory const*>(&(mf.Factory()));
    if (factory == nullptr) return;
    const auto& flags = factory->getMultiEBCellFlagFab();

    AMREX_ALWAYS_ASSERT(mf.ixType().cellCentered() || mf.ixType().nodeCentered());
    bool is_cell_centered = mf.ixType().cellCentered();
    int ng = std::min(mf.nGrow(),ngrow);

    Gpu::AsyncArray<Real> vals_aa(a_vals.data(), ncomp);
    Real const* AMREX_RESTRICT vals = vals_aa.data();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);
        const auto& flagarr = flags.const_array(mfi);
        Array4<Real> const& arr = mf.array(mfi);

        if (is_cell_centered) {
            AMREX_HOST_DEVICE_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                if (flagarr(i,j,k).isCovered()) {
                    arr(i,j,k,n+icomp) = vals[n];
                }
            });
        } else {
            AMREX_HOST_DEVICE_FOR_4D (bx, ncomp, i, j, k, n,
            {
                eb_set_covered_nodes(i,j,k,n,icomp,arr,flagarr,vals);
            });
        }
    }
}

void
EB_set_covered_faces (const Array<MultiFab*,AMREX_SPACEDIM>& umac, Real val)
{
    const auto factory = dynamic_cast<EBFArrayBoxFactory const*>(&(umac[0]->Factory()));
    if (factory == nullptr) return;

    const auto& area = factory->getAreaFrac();
    const auto& flags = factory->getMultiEBCellFlagFab();
    const int ncomp = umac[0]->nComp();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*umac[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        AMREX_D_TERM(const Box& xbx = mfi.tilebox(IntVect::TheDimensionVector(0));,
                     const Box& ybx = mfi.tilebox(IntVect::TheDimensionVector(1));,
                     const Box& zbx = mfi.tilebox(IntVect::TheDimensionVector(2)););
        AMREX_D_TERM(Array4<Real> const& u = umac[0]->array(mfi);,
                     Array4<Real> const& v = umac[1]->array(mfi);,
                     Array4<Real> const& w = umac[2]->array(mfi););

        auto fabtyp = flags[mfi].getType();
        if (fabtyp == FabType::covered)
        {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
                xbx, txbx,
                {
                    const auto lo = amrex::lbound(txbx);
                    const auto hi = amrex::ubound(txbx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            u(i,j,k,n) = 0.0;
                        }}}
                    }
                }
                ,ybx, tybx,
                {
                    const auto lo = amrex::lbound(tybx);
                    const auto hi = amrex::ubound(tybx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            v(i,j,k,n) = 0.0;
                        }}}
                    }
                }
#if (AMREX_SPACEDIM == 3)
                ,zbx, tzbx,
                {
                    const auto lo = amrex::lbound(tzbx);
                    const auto hi = amrex::ubound(tzbx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            w(i,j,k,n) = 0.0;
                        }}}
                    }
                }
#endif
                );
        }
        else if (fabtyp == FabType::singlevalued)
        {
            AMREX_D_TERM(Array4<Real const> const& ax = area[0]->const_array(mfi);,
                         Array4<Real const> const& ay = area[1]->const_array(mfi);,
                         Array4<Real const> const& az = area[2]->const_array(mfi););
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA (
                xbx, txbx,
                {
                    const auto lo = amrex::lbound(txbx);
                    const auto hi = amrex::ubound(txbx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            if (ax(i,j,k) == 0.0) u(i,j,k,n) = 0.0;
                        }}}
                    }
                }
                ,ybx, tybx,
                {
                    const auto lo = amrex::lbound(tybx);
                    const auto hi = amrex::ubound(tybx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            if (ay(i,j,k) == 0.0) v(i,j,k,n) = 0.0;
                        }}}
                    }
                }
#if (AMREX_SPACEDIM == 3)
                ,zbx, tzbx,
                {
                    const auto lo = amrex::lbound(tzbx);
                    const auto hi = amrex::ubound(tzbx);
                    for (int n = 0; n < ncomp; ++n) {
                        for (int k = lo.z; k <= hi.z; ++k) {
                        for (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                            if (az(i,j,k) == 0.0) w(i,j,k,n) = 0.0;
                        }}}
                    }
                }
#endif
                );
        }
    }
}

void
EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, const MultiFab& vol_fine,
                 const MultiFab& vfrac_fine, int scomp, int ncomp, const IntVect& ratio)
{
    BL_PROFILE("EB_average_down");

    AMREX_ASSERT(S_fine.ixType().cellCentered());
    AMREX_ASSERT(S_crse.ixType().cellCentered());

    const DistributionMapping& fine_dm = S_fine.DistributionMap();
    BoxArray crse_S_fine_BA = S_fine.boxArray();
    crse_S_fine_BA.coarsen(ratio);

    MultiFab crse_S_fine(crse_S_fine_BA,fine_dm,ncomp,0,MFInfo(),FArrayBoxFactory());

    Dim3 dratio = ratio.dim3();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tbx = mfi.tilebox();
        auto& crse_fab = crse_S_fine[mfi];
        const auto& fine_fab = S_fine[mfi];

        const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
        FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

        Array4<Real> const& crse_arr = crse_fab.array();
        Array4<Real const> const& fine_arr = fine_fab.const_array();
        Array4<Real const> const& vol = vol_fine.const_array(mfi);

        if (typ == FabType::regular || typ == FabType::covered)
        {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
            {
                amrex_avgdown_with_vol(b, crse_arr, fine_arr, vol, 0, scomp, ncomp, ratio);
            });
        }
        else if (typ == FabType::singlevalued)
        {
            Array4<Real const> const& vfrac = vfrac_fine.const_array(mfi);
            AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
            {
                eb_avgdown_with_vol(i,j,k,fine_arr,scomp,crse_arr,0,vol,vfrac,dratio,ncomp);
            });
        }
        else
        {
            amrex::Abort("multi-valued avgdown to be implemented");
        }
    }

    S_crse.copy(crse_S_fine,0,scomp,ncomp);
}


void
EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp, int ratio)
{
    EB_average_down(S_fine, S_crse, scomp, ncomp, IntVect(ratio));
}

void
EB_average_down (const MultiFab& S_fine, MultiFab& S_crse, int scomp, int ncomp, const IntVect& ratio)
{
    if (!S_fine.hasEBFabFactory())
    {
        amrex::average_down(S_fine, S_crse, scomp, ncomp, ratio);
    }
    else
    {
        Dim3 dratio = ratio.dim3();

        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(S_fine.Factory());
        const auto& vfrac_fine = factory.getVolFrac();

        BL_ASSERT(S_crse.nComp() == S_fine.nComp());
        BL_ASSERT(S_crse.is_cell_centered() && S_fine.is_cell_centered());

        BoxArray crse_S_fine_BA = S_fine.boxArray(); crse_S_fine_BA.coarsen(ratio);

        if (crse_S_fine_BA == S_crse.boxArray()
            and S_fine.DistributionMap() == S_crse.DistributionMap())
        {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(S_crse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& tbx = mfi.tilebox();
                auto& crse_fab = S_crse[mfi];
                const auto& fine_fab = S_fine[mfi];

                const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
                FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

                Array4<Real> const& crse = crse_fab.array();
                Array4<Real const> const& fine = fine_fab.const_array();

                if (typ == FabType::regular || typ == FabType::covered)
                {
                    AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
                    {
                        amrex_avgdown(b,crse,fine,scomp,scomp,ncomp,ratio);
                    });
                }
                else
                {
                    Array4<Real const> const& vfrc = vfrac_fine.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
                    {
                        eb_avgdown(i,j,k,fine,scomp,crse,scomp,vfrc,dratio,ncomp);
                    });
                }
            }
        }
        else
        {
            MultiFab crse_S_fine(crse_S_fine_BA, S_fine.DistributionMap(),
                                 ncomp, 0, MFInfo(),FArrayBoxFactory());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(crse_S_fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& tbx = mfi.tilebox();
                auto& crse_fab = crse_S_fine[mfi];
                const auto& fine_fab = S_fine[mfi];

                const auto& flag_fab = amrex::getEBCellFlagFab(fine_fab);
                FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

                Array4<Real> const& crse_arr = crse_fab.array();
                Array4<Real const> const& fine_arr = fine_fab.const_array();

                if (typ == FabType::regular || typ == FabType::covered)
                {
                    AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
                    {
                        amrex_avgdown(b,crse_arr,fine_arr,0,scomp,ncomp,ratio);
                    });
                }
                else if (typ == FabType::singlevalued)
                {
                    Array4<Real const> const& vfrc = vfrac_fine.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(tbx, i, j, k,
                    {
                        eb_avgdown(i,j,k,fine_arr,scomp,crse_arr,scomp,vfrc,dratio,ncomp);
                    });
                }
                else
                {
                    amrex::Abort("multi-valued avgdown to be implemented");
                }
            }

            S_crse.copy(crse_S_fine,0,scomp,ncomp);
        }
    }
}


void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
                            const Array<MultiFab*,AMREX_SPACEDIM>& crse,
                            int ratio, int ngcrse)
{
    EB_average_down_faces(fine, crse, IntVect{ratio}, ngcrse);
}

void EB_average_down_faces (const Array<const MultiFab*,AMREX_SPACEDIM>& fine,
                            const Array<MultiFab*,AMREX_SPACEDIM>& crse,
                            const IntVect& ratio, int ngcrse)
{
    AMREX_ASSERT(crse[0]->nComp() == fine[0]->nComp());

    int ncomp = crse[0]->nComp();
    if (!(*fine[0]).hasEBFabFactory())
    {
        amrex::average_down_faces(fine, crse, ratio, ngcrse);
    }
    else
    {
        Dim3 dratio = ratio.dim3();

        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>((*fine[0]).Factory());
        const auto&  aspect = factory.getAreaFrac();

        if (isMFIterSafe(*fine[0], *crse[0]))
        {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (int n=0; n<AMREX_SPACEDIM; ++n) {
                for (MFIter mfi(*crse[n],TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const auto& flag_fab = amrex::getEBCellFlagFab((*fine[n])[mfi]);
                    const Box& tbx = mfi.growntilebox(ngcrse);
                    FabType typ = flag_fab.getType(amrex::refine(tbx,ratio));

                    Array4<Real> const& ca = crse[n]->array(mfi);
                    Array4<Real const> const& fa = fine[n]->const_array(mfi);

                    if(typ == FabType::regular || typ == FabType::covered)
                    {
                        AMREX_LAUNCH_HOST_DEVICE_LAMBDA(tbx, b,
                        {
                            amrex_avgdown_faces(b, ca, fa, 0, 0, ncomp, ratio, n);
                        });
                    }
                    else
                    {
                        Array4<Real const> const& ap = aspect[n]->const_array(mfi);
                        if (n == 0) {
                            AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
                            {
                                eb_avgdown_face_x(i,j,k,fa,0,ca,0,ap,dratio,ncomp);
                            });
                        } else if (n == 1) {
                            AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
                            {
                                eb_avgdown_face_y(i,j,k,fa,0,ca,0,ap,dratio,ncomp);
                            });
                        } else {
#if (AMREX_SPACEDIM == 3)
                            AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
                            {
                                eb_avgdown_face_z(i,j,k,fa,0,ca,0,ap,dratio,ncomp);
                            });
#endif
                        }
                    }
                }
            }
        }
        else
        {
            Array<MultiFab,AMREX_SPACEDIM> ctmp;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                BoxArray cba = fine[idim]->boxArray();
                cba.coarsen(ratio);
                ctmp[idim].define(cba, fine[idim]->DistributionMap(), ncomp, ngcrse, MFInfo(), FArrayBoxFactory());
            }
            EB_average_down_faces(fine, amrex::GetArrOfPtrs(ctmp), ratio, ngcrse);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                crse[idim]->ParallelCopy(ctmp[idim],0,0,ncomp,ngcrse,ngcrse);
            }
        }
    }
}

void EB_average_down_boundaries (const MultiFab& fine, MultiFab& crse,
                                 int ratio, int ngcrse)
{
    EB_average_down_boundaries(fine,crse,IntVect{ratio},ngcrse);
}

void EB_average_down_boundaries (const MultiFab& fine, MultiFab& crse,
                                 const IntVect& ratio, int ngcrse)
{
    int ncomp = crse.nComp();

    if (!fine.hasEBFabFactory())
    {
        crse.setVal(0.0, 0, ncomp, ngcrse);
    }
    else
    {
        Dim3 dratio = ratio.dim3();

        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(fine.Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& barea = factory.getBndryArea();

        if (isMFIterSafe(fine, crse))
        {
            MFItInfo info;
            if (Gpu::notInLaunchRegion()) info.EnableTiling().SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(crse, info); mfi.isValid(); ++mfi)
            {
                const Box& tbx = mfi.growntilebox(ngcrse);
                FabType typ = flags[mfi].getType(amrex::refine(tbx,ratio));
                Array4<Real> const& ca = crse.array(mfi);
                if (FabType::covered == typ || FabType::regular == typ) {
                    AMREX_HOST_DEVICE_FOR_4D(tbx,ncomp,i,j,k,n,
                    {
                        ca(i,j,k,n) = 0.0;
                    });
                } else {
                    Array4<Real const> const& fa = fine.const_array(mfi);
                    Array4<Real const> const& ba = barea.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(tbx,i,j,k,
                    {
                        eb_avgdown_boundaries(i,j,k,fa,0,ca,0,ba,dratio,ncomp);
                    });
                }
            }
        }
        else
        {
            BoxArray cba = fine.boxArray();
            cba.coarsen(ratio);
            MultiFab ctmp(cba, fine.DistributionMap(), ncomp, ngcrse, MFInfo(), FArrayBoxFactory());
            EB_average_down_boundaries(fine, ctmp, ratio, ngcrse);
            crse.ParallelCopy(ctmp, 0, 0, ncomp, ngcrse, ngcrse);
        }
    }
}


void EB_computeDivergence (MultiFab& divu, const Array<MultiFab const*,AMREX_SPACEDIM>& umac,
                           const Geometry& geom, bool already_on_centroids)
{
    AMREX_ASSERT(divu.nComp()==umac[0]->nComp());
    AMREX_ASSERT(divu.nComp()==umac[1]->nComp());
#if (AMREX_SPACEDIM == 3)
    AMREX_ASSERT(divu.nComp()==umac[2]->nComp());
#endif
    
    if (!divu.hasEBFabFactory())
    {
        amrex::computeDivergence(divu, umac, geom);
    }
    else
    {
        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(divu.Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& vfrac = factory.getVolFrac();
        const auto& area = factory.getAreaFrac();
        const auto& fcent = factory.getFaceCent();

        iMultiFab cc_mask;
        if (!already_on_centroids) {
            cc_mask.define(divu.boxArray(), divu.DistributionMap(), 1, 1);
            cc_mask.BuildMask(geom.Domain(), geom.periodicity(), 1, 0, 0, 1);
        }

        const GpuArray<Real,AMREX_SPACEDIM> dxinv = geom.InvCellSizeArray();
        MFItInfo info;
        if (Gpu::notInLaunchRegion()) info.EnableTiling().SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(divu,info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto& flagfab = flags[mfi];
            Array4<Real> const& divuarr = divu.array(mfi);
            AMREX_D_TERM(Array4<Real const> const& uarr = umac[0]->const_array(mfi);,
                         Array4<Real const> const& varr = umac[1]->const_array(mfi);,
                         Array4<Real const> const& warr = umac[2]->const_array(mfi););

            const auto fabtyp = flagfab.getType(bx);
            if (fabtyp == FabType::covered) {
                AMREX_HOST_DEVICE_FOR_4D(bx,divu.nComp(),i,j,k,n,
                {
                    divuarr(i,j,k,n) = 0.0;
                });
            } else if (fabtyp == FabType::regular) {
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA(bx, b,
                {
                    amrex_compute_divergence(b,divuarr,AMREX_D_DECL(uarr,varr,warr),dxinv);
                });
            } else {
                Array4<int const> const& ccm = (already_on_centroids) ?
                    Array4<int const>{} : cc_mask.const_array(mfi);
                Array4<Real const> const& vol = vfrac.const_array(mfi);
                AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
                             Array4<Real const> const& apy = area[1]->const_array(mfi);,
                             Array4<Real const> const& apz = area[2]->const_array(mfi););
                AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                             Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                             Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
                Array4<EBCellFlag const> const& flagarr = flagfab.const_array();
                AMREX_HOST_DEVICE_FOR_4D(bx,divu.nComp(),i,j,k,n,
                {
                    eb_compute_divergence(i,j,k,n,divuarr,AMREX_D_DECL(uarr,varr,warr),
                                          ccm, flagarr, vol, AMREX_D_DECL(apx,apy,apz),
                                          AMREX_D_DECL(fcx,fcy,fcz), dxinv, already_on_centroids);
                });
            }
        }
    }
}

void
EB_average_face_to_cellcenter (MultiFab& ccmf, int dcomp,
                               const Array<MultiFab const*,AMREX_SPACEDIM>& fmf)
{
    AMREX_ASSERT(ccmf.nComp() >= dcomp + AMREX_SPACEDIM);

    if (!fmf[0]->hasEBFabFactory())
    {
        average_face_to_cellcenter(ccmf, dcomp, fmf);
    }
    else
    {
        const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(fmf[0]->Factory());
        const auto& flags = factory.getMultiEBCellFlagFab();
        const auto& area = factory.getAreaFrac();

        MFItInfo info;
        if (Gpu::notInLaunchRegion()) info.EnableTiling().SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(ccmf,info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto& flagfab = flags[mfi];
            Array4<Real> const& ccfab = ccmf.array(mfi);
            AMREX_D_TERM(Array4<Real const> const& xfab = fmf[0]->const_array(mfi);,
                         Array4<Real const> const& yfab = fmf[1]->const_array(mfi);,
                         Array4<Real const> const& zfab = fmf[2]->const_array(mfi););
            const auto fabtyp = flagfab.getType(bx);
            if (fabtyp == FabType::covered) {
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    ccfab(i,j,k,dcomp) = 0.0;
                });
            } else if (fabtyp == FabType::regular) {
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA(bx, b,
                {
                    amrex_avg_fc_to_cc(b,ccfab,AMREX_D_DECL(xfab,yfab,zfab),dcomp);
                });
            } else {
                AMREX_D_TERM(Array4<Real const> const& apx = area[0]->const_array(mfi);,
                             Array4<Real const> const& apy = area[1]->const_array(mfi);,
                             Array4<Real const> const& apz = area[2]->const_array(mfi););
                Array4<EBCellFlag const> const& flagarr = flagfab.const_array();
                AMREX_HOST_DEVICE_FOR_3D(bx,i,j,k,
                {
                    eb_avg_fc_to_cc(i,j,k,dcomp,ccfab,AMREX_D_DECL(xfab,yfab,zfab),
                                    AMREX_D_DECL(apx,apy,apz),flagarr);
                });
            }
        }
    }
}

void
EB_interp_CC_to_Centroid (MultiFab& cent, const MultiFab& cc, int scomp, int dcomp, int ncomp, const Geometry& geom)
{
    const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(cc.Factory());
    const auto& flags = factory.getMultiEBCellFlagFab();
    const auto& loc = factory.getCentroid();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(cc, mfi_info);  mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        const auto& centfab = cent.array(mfi,dcomp);
        const auto& fabtyp = flags[mfi].getType(vbx);

        if (fabtyp == FabType::covered)
        {
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( vbx, ncomp, i, j, k, n,
            {
                centfab(i,j,k,n) = 0.0;
            });
        }
        else if (fabtyp == FabType::regular)
        {
            const auto& ccfab = cc.array(mfi,scomp);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( vbx, ncomp, i, j, k, n,
            {
               centfab(i,j,k,n) = ccfab(i,j,k,n);
            });
        }
        else
        {
            const auto& flagfab = flags.const_array(mfi);
            const auto& locfab = loc.const_array(mfi);
            const auto& ccfab = cc.array(mfi,scomp);

            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
            {
                eb_interp_cc2cent(thread_box, centfab, ccfab, flagfab, locfab, ncomp);
            });
        }
    }

    cent.FillBoundary(dcomp,ncomp,geom.periodicity());
    
}

void
EB_interp_CC_to_FaceCentroid (MultiFab& sol,
                              AMREX_D_DECL( MultiFab& edgestate_x,
                                            MultiFab& edgestate_y,
                                            MultiFab& edgestate_z),
                              int scomp, int dcomp, int ncomp,
                              const Geometry& a_geom,
                              const Vector<BCRec>& a_bcs)
{
    const auto& factory = dynamic_cast<EBFArrayBoxFactory const&>(sol.Factory());
    const auto& flags = factory.getMultiEBCellFlagFab();
    const auto& area = factory.getAreaFrac();
    const auto& fcent = factory.getFaceCent();

    AMREX_ALWAYS_ASSERT(a_bcs.size() == ncomp );
    
    Box domain(a_geom.Domain());
    
    const int nghost(4);
    
   // Initialize edge state
    AMREX_D_TERM(edgestate_x.setVal(1e40,dcomp,ncomp);,
                 edgestate_y.setVal(1e40,dcomp,ncomp);,
                 edgestate_z.setVal(1e40,dcomp,ncomp));

    BCRec const* d_bcs;
#ifdef AMREX_USE_GPU
    Gpu::DeviceVector<BCRec> dv_bcs(ncomp);
    if (Gpu::inLaunchRegion())
    {
        Gpu::copy(Gpu::hostToDevice, a_bcs.begin(), a_bcs.begin()+ncomp, dv_bcs.begin());
        d_bcs = dv_bcs.dataPtr();
    }
    else
#endif
    {
        d_bcs = a_bcs.dataPtr();
    }
    
    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol, mfi_info);  mfi.isValid(); ++mfi)
    {

        const Box& vbx = mfi.tilebox();
        const auto& solnfab = sol.array(mfi,scomp);
        const auto fabtyp = flags[mfi].getType(amrex::grow(vbx,0));
        const auto fabtyp_ghost = flags[mfi].getType(amrex::grow(vbx,nghost));

        if (fabtyp != FabType::covered)
        {

            AMREX_D_TERM(Array4<Real> const& edg_x = edgestate_x.array(mfi,dcomp);,
                         Array4<Real> const& edg_y = edgestate_y.array(mfi,dcomp);,
                         Array4<Real> const& edg_z = edgestate_z.array(mfi,dcomp););
          
            if (fabtyp_ghost == FabType::regular )
            {
                
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
                {
                    eb_interp_cc2face(thread_box, solnfab,
                                      AMREX_D_DECL(edg_x,edg_y,edg_z),
                                      ncomp,
                                      domain, d_bcs);
                });
            
            }
            else
            {
                Array4<EBCellFlag const> const& flagfab = flags.const_array(mfi);
                AMREX_D_TERM(Array4<Real const> const& apxfab = area[0]->const_array(mfi);,
                             Array4<Real const> const& apyfab = area[1]->const_array(mfi);,
                             Array4<Real const> const& apzfab = area[2]->const_array(mfi););
                AMREX_D_TERM(Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                             Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                             Array4<Real const> const& fcz = fcent[2]->const_array(mfi););
                
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( vbx, thread_box,
                {
                    eb_interp_cc2facecent(thread_box, solnfab,
                                          flagfab,
                                          AMREX_D_DECL(apxfab,apyfab,apzfab),
                                          AMREX_D_DECL(fcx,fcy,fcz),
                                          AMREX_D_DECL(edg_x,edg_y,edg_z),
                                          ncomp,
                                          domain, d_bcs);
                });
            }
        }
    }
    
    edgestate_x.FillBoundary(a_geom.periodicity());
    edgestate_y.FillBoundary(a_geom.periodicity());
#if ( AMREX_SPACEDIM == 3 )
    edgestate_z.FillBoundary(a_geom.periodicity());
#endif

}

}
