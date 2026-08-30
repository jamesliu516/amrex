#ifndef WMLES_SGS_K_H_
#define WMLES_SGS_K_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_EBCellFlag.H>
#include "WMLES_parm.H"

// ---------------------------------------------------------------------------
// Compute Smagorinsky eddy viscosity nu_t = (Cs * Delta)^2 * |S|
// Delta = filter_width_ratio * (dx*dy*dz)^(1/3)
// |S|   = sqrt(2 * S_ij * S_ij)
// ---------------------------------------------------------------------------

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
wmles_compute_nu_t (int i, int j, int k,
                    amrex::Array4<amrex::Real const> const& q,
                    amrex::Array4<amrex::EBCellFlag const> const& flag,
                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
                    Parm const& parm) noexcept
{
    using amrex::Real;

    if (flag(i,j,k).isCovered()) { return Real(0.0); }

    // Velocity gradients at cell center (2nd-order central difference)
    AMREX_D_TERM(
    Real dudx = (q(i+1,j,k,QU) - q(i-1,j,k,QU)) * Real(0.5) * dxinv[0];,
    Real dudy = (q(i,j+1,k,QU) - q(i,j-1,k,QU)) * Real(0.5) * dxinv[1];,
    Real dudz = (q(i,j,k+1,QU) - q(i,j,k-1,QU)) * Real(0.5) * dxinv[2];)

    AMREX_D_TERM(
    Real dvdx = (q(i+1,j,k,QV) - q(i-1,j,k,QV)) * Real(0.5) * dxinv[0];,
    Real dvdy = (q(i,j+1,k,QV) - q(i,j-1,k,QV)) * Real(0.5) * dxinv[1];,
    Real dvdz = (q(i,j,k+1,QV) - q(i,j,k-1,QV)) * Real(0.5) * dxinv[2];)

#if (AMREX_SPACEDIM == 3)
    Real dwdx = (q(i+1,j,k,QW) - q(i-1,j,k,QW)) * Real(0.5) * dxinv[0];
    Real dwdy = (q(i,j+1,k,QW) - q(i,j-1,k,QW)) * Real(0.5) * dxinv[1];
    Real dwdz = (q(i,j,k+1,QW) - q(i,j,k-1,QW)) * Real(0.5) * dxinv[2];
#endif

    // Strain rate magnitude squared: 2 * S_ij * S_ij
#if (AMREX_SPACEDIM == 2)
    Real S2 = Real(2.0)*dudx*dudx + Real(2.0)*dvdy*dvdy
            + (dudy+dvdx)*(dudy+dvdx);
#else
    Real S2 = Real(2.0)*(dudx*dudx + dvdy*dvdy + dwdz*dwdz)
            + (dudy+dvdx)*(dudy+dvdx)
            + (dudz+dwdx)*(dudz+dwdx)
            + (dvdz+dwdy)*(dvdz+dwdy);
#endif

    Real strate = std::sqrt(S2);

    // Filter width: Delta = (dx*dy*dz)^(1/3) * filter_width_ratio
    Real Delta = parm.sgs_filter_width_ratio;
#if (AMREX_SPACEDIM == 2)
    Delta *= std::sqrt(Real(1.0) / (dxinv[0]*dxinv[1]));
#else
    Delta *= std::cbrt(Real(1.0) / (dxinv[0]*dxinv[1]*dxinv[2]));
#endif

    Real nu_t = (parm.Cs_smag * Delta) * (parm.Cs_smag * Delta) * strate;

    return nu_t;
}

#endif // WMLES_SGS_K_H_
