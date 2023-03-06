# /usr/bin/env python
"""
Python code taken originally from:

```bibtex
    @misc { PURR4066,
	title = {Code for tropical cyclone wind profile model of Chavas et al (2015, JAS)},
	month = {Jun},
	url = {https://purr.purdue.edu/publications/4066/1},
	year = {2022},
	doi = {doi:/10.4231/CZ4P-D448},
	author = {Daniel Robert Chavas ,}
}
```

Python code originally written in python2 by Chia-Ying Lee.

Python code converted to python3 by Simon Thomas.

Reference publications:

Chavas, D. R, N. Lin, and K. A. Emanuel (2015).
A complete tropical cyclone radial wind structure model.
Part I: Comparison with observed structure. J. Atmos. Sci., 72(9):
3647-3662. doi:10.1175/JAS-D-15-0014.1.

Chavas, D. R. and Lin, N. (2016).
A model for the complete radial structure of the tropical cyclone wind field.
Part II: Wind field variability. J. Atmos. Sci., 73(8):
3093-3113. doi:10.1175/JAS-D-15-0185.1.

# TODO: work out if units are SI or not
"""
from typing import Tuple, List, Dict, Union
import numpy as np
import sys
import copy
from shapely.geometry import LineString
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

#######################################################################
# NOTES FOR USER:
# Parameter units listed in []
# Characteristic values listed in {}
#######################################################################


def E04_outerwind_r0input_nondim_MM0(
    r0: float, fcor: float, Cdvary: float, C_d: float, w_cool: float, Nr: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E04_outerwind_r0input_nondim_MM0

    Args:
        r0 (float): outer radius of storm.
        fcor (float): coriolis parameter.
        Cdvary (float): _description_
        C_d (float): _description_
        w_cool (float): _description_
        Nr (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

    # Initialization
    fcor = abs(fcor)
    M0 = 0.5 * fcor * r0**2  # [m2/s] M at outer radius

    drfracr0 = 0.001
    # I replaced a binary or `|` with an `or` as I thought it was more expressive.
    if (r0 > 2500 * 1000) or (r0 < 200 * 1000):
        drfracr0 = drfracr0 / 10
        # extra precision for very large storm to avoid funny bumps near r0 (though rest of solution is stable!)
        # or for tiny storm that requires E04 extend to very small radii to
        # match with ER11

    if Nr > 1 / drfracr0:
        Nr = 1 / drfracr0  # grid radii must be > 0

    rfracr0_max = 1  # [-] start at r0, move radially inwards
    rfracr0_min = rfracr0_max - (Nr - 1) * drfracr0  # [-] inner-most node
    rrfracr0 = np.arange(
        rfracr0_min, rfracr0_max + drfracr0, drfracr0
    )  # [] r/r0 vector
    # [] M/M0 vector initialized to 1 (M/M0 = 1 at r/r0=1)
    MMfracM0 = np.float("NaN") * np.zeros(rrfracr0.size)
    MMfracM0[-1] = 1

    # First step inwards from r0: d(M/M0)/d(r/r0) = 0 by definition
    rfracr0_temp = rrfracr0[-2]  # one step inwards from r0
    # dMfracM0_drfracr0_temp = 0    #[] d(M/M0)/d(r/r0) = 0 at r/r0 = 1
    MfracM0_temp = MMfracM0[-1]
    MMfracM0[-2] = MfracM0_temp
    ##################################################################
    ## Variable C_d: code from Cd_Donelan04.m (function call is slow) ######
    # Piecewise linear fit parameters estimated from Donelan2004_fit.m
    C_d_lowV = 6.2e-4
    V_thresh1 = 6  # m/s transition from constant to linear increasing
    V_thresh2 = 35.4  # m/s transition from linear increasing to constant
    C_d_highV = 2.35e-3
    linear_slope = (C_d_highV - C_d_lowV) / (V_thresh2 - V_thresh1)
    ##################################################################

    # Integrate inwards from r0 to obtain profile of M/M0 vs. r/r0
    for ii in range(0, np.int(Nr) - 2, 1):  # first two nodes already done above

        # Calculate C_d varying with V, if desired
        if Cdvary == 1:

            # Calculate V at this r/r0 (for variable C_d only)
            V_temp = (M0 / r0) * ((MfracM0_temp / rfracr0_temp) - rfracr0_temp)

            # Calculate C_d
            if V_temp <= V_thresh1:
                C_d = C_d_lowV
            elif V_temp > V_thresh2:
                C_d = C_d_highV
            else:
                C_d = C_d_lowV + linear_slope * (V_temp - V_thresh1)

        # Calculate model parameter, gamma
        gam = C_d * fcor * r0 / w_cool  # [] non-dimensional model parameter

        # Update dMfracM0_drfracr0 at next step inwards
        dMfracM0_drfracr0_temp = (
            gam * ((MfracM0_temp - rfracr0_temp**2) ** 2) / (1 - rfracr0_temp**2)
        )

        # Integrate M/M0 radially inwards
        MfracM0_temp = MfracM0_temp - dMfracM0_drfracr0_temp * drfracr0

        # Update r/r0 to follow M/M0
        rfracr0_temp = rfracr0_temp - drfracr0  # [] move one step inwards

        # Save updated values
        MMfracM0[MMfracM0.shape[0] - 1 - ii - 2] = MfracM0_temp

    return rrfracr0, MMfracM0


def ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11):
    fcor = np.abs(fcor)
    if rmax_or_r0 == "rmax":
        rmax = r_in
    else:
        print('rmax_or_r0 must be set to"rmax"')
    # CALCULATE Emanuel and Rotunno (2011) theoretical profile
    V_ER11 = (1.0 / rr_ER11) * (Vmax * rmax + 0.5 * fcor * rmax**2) * (
        (2 * (rr_ER11 / rmax) ** 2) / (2 - CkCd + CkCd * (rr_ER11 / rmax) ** 2)
    ) ** (1 / (2 - CkCd)) - 0.5 * fcor * rr_ER11
    # make V=0 at r=0
    V_ER11[rr_ER11 == 0] = 0

    if rmax_or_r0 == "rmax":
        i_rmax = np.argwhere(V_ER11 == np.max(V_ER11))[0, 0]
        f = interp1d(
            V_ER11[i_rmax + 1 :], rr_ER11[i_rmax + 1 :], fill_value="extrapolate"
        )
        r0_profile = f(0.0)
        r_out = r0_profile.tolist()  # use value from profile itself
    else:
        print('rmax_or_r0 must be set to"rmax"')
    return V_ER11, r_out


def ER11_radprof(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11):
    dr = rr_ER11[1] - rr_ER11[0]
    # Call ER11_radprof_raw
    V_ER11, r_out = ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11)
    if rmax_or_r0 == "rmax":
        drin_temp = r_in - rr_ER11[np.argwhere(V_ER11 == np.max(V_ER11))[0, 0]]
    elif rmax_or_r0 == "r0":
        f = interp1d(V_ER11[2:], rr_ER11[2:])
        drin_temp = r_in - f(0).tolist()
    # Calculate error in Vmax
    dVmax_temp = Vmax - np.max(V_ER11)

    # Check is errors are too large and adjust accordingly
    r_in_save = copy.copy(r_in)
    Vmax_save = copy.copy(Vmax)

    n_iter = 0
    # if error is sufficiently large NOTE: FIRST ARGUMENT MUST BE ">" NOT ">="
    # or else rmax values at exactly dr/2 intervals (e.g. 10.5 for dr=1 km)
    # will not converge
    while (np.abs(drin_temp) > dr / 2) or (np.abs(dVmax_temp / Vmax_save) >= 10**-2):

        # drin_temp/1000

        n_iter = n_iter + 1
        if n_iter > 20:
            # sprintf('ER11 CALCULATION DID NOT CONVERGE TO INPUT (RMAX,VMAX) =
            # (#3.1f km,#3.1f m/s) Ck/Cd =
            # #2.2f!',r_in_save/1000,Vmax_save,CkCd)
            V_ER11 = np.float("NaN") * np.zeros(rr_ER11.size)
            r_out = np.float("NaN")
            break

        # Adjust estimate of r_in according to error
        r_in = r_in + drin_temp

        # Vmax second
        while np.abs(dVmax_temp / Vmax) >= 10**-2:  # if error is sufficiently large

            #                    dVmax_temp

            # Adjust estimate of Vmax according to error
            Vmax = Vmax + dVmax_temp

            [V_ER11, r_out] = ER11_radprof_raw(
                Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11
            )
            Vmax_prof = np.max(V_ER11)
            dVmax_temp = Vmax_save - Vmax_prof

        [V_ER11, r_out] = ER11_radprof_raw(Vmax, r_in, rmax_or_r0, fcor, CkCd, rr_ER11)
        Vmax_prof = np.max(V_ER11)
        dVmax_temp = Vmax_save - Vmax_prof
        if rmax_or_r0 == "rmax":
            drin_temp = r_in_save - rr_ER11[np.argwhere(V_ER11 == Vmax_prof)[0, 0]]
        elif rmax_or_r0 == "r0":
            f = interp1d(V_ER11[2:], rr_ER11[2:])
            drin_temp = r_in_save - f(0)

    return V_ER11, r_out


def ER11E04_nondim_r0input(
    Vmax,
    r0,
    fcor,
    Cdvary,
    C_d,
    w_cool,
    CkCdvary: float = 1,
    CkCd: float = 1.9,
    eye_adj: float = 1,
    alpha_eye: float = 1,
):

    # Initialization
    fcor = np.abs(fcor)
    if CkCdvary == 1:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst
    # 'Ck/Cd is capped at 1.9 and has been set to this value. If CkCdvary=1,
    # then Vmax is much greater than the range of data used to estimate
    # CkCd as a function of Vmax -- here be dragons!')
    CkCd = np.min((1.9, CkCd))

    # Step 1: Calculate E04 M/M0 vs. r/r0
    Nr = 100000
    rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(
        r0, fcor, Cdvary, C_d, w_cool, Nr
    )

    M0_E04 = 0.5 * fcor * r0**2

    # Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point
    # with E04 M/M0

    count = 0
    soln_converged = 0
    while soln_converged == 0:
        count += 1
        print(count)
        # Break up interval into 3 points, take 2 between which intersection
        # vanishes, repeat til converges
        rmaxr0_min = 0.001
        rmaxr0_max = 0.75
        rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2.0  # first guess -- in the middle
        rmaxr0 = rmaxr0_new  # initialize
        drmaxr0 = rmaxr0_max - rmaxr0  # initialize
        drmaxr0_thresh = 0.000001
        iterN = 0
        rfracrm_min = 0.0  # [-] start at r=0
        rfracrm_max = 50.0  # [-] extend out to many rmaxs
        while np.abs(drmaxr0) >= drmaxr0_thresh:
            iterN = iterN + 1
            # Calculate ER11 M/Mm vs r/rm
            rmax = rmaxr0_new * r0  # [m]
            print(rmax)
            #         [~,~,rrfracrm_ER11,MMfracMm_ER11] =
            #         ER11_radprof_nondim(Vmax,rmax,fcor,CkCd) #FAILS FOR LOW CK/CD NOT
            #         SURE WHY
            drfracrm = 0.01
            if rmax > 100.0 * 1000:
                drfracrm = drfracrm / 10.0  # extra precision for large storm

            rrfracrm_ER11 = np.arange(
                rfracrm_min, rfracrm_max + drfracrm, drfracrm
            )  # [] r/r0 vector
            rr_ER11 = rrfracrm_ER11 * rmax
            rmax_or_r0 = "rmax"
            VV_ER11, dummy = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)

            if not np.isnan(np.max(VV_ER11)):  # ER11_radprof converged
                rrfracr0_ER11 = rr_ER11 / r0
                MMfracM0_ER11 = (rr_ER11 * VV_ER11 + 0.5 * fcor * rr_ER11**2) / M0_E04
                l1 = LineString(list(zip(rrfracr0_E04, MMfracM0_E04)))
                l2 = LineString(list(zip(rrfracr0_ER11, MMfracM0_ER11)))
                intersection = l1.intersection(l2)
                if (
                    intersection.wkt == "GEOMETRYCOLLECTION EMPTY"
                ):  # no intersections -- rmaxr0 too small
                    drmaxr0 = np.abs(drmaxr0) / 2
                else:
                    if intersection.wkt.split(" ")[0] == "POINT":
                        X0, Y0 = intersection.coords[0]
                    elif intersection.wkt.split(" ")[0] == "MULTIPOINT":
                        X0, Y0 = intersection[0].coords[0]
                    # at least one intersection -- rmaxr0 too large
                    drmaxr0 = -np.abs(drmaxr0) / 2
                    rmerger0 = np.mean(X0)
                    MmergeM0 = np.mean(Y0)
            # ER11_radprof did not converge -- convergence fails for low CkCd
            # and high Ro = Vm/(f*rm)
            else:
                # Must reduce rmax (and thus reduce Ro)
                drmaxr0 = -abs(drmaxr0) / 2
            # update value of rmaxr0
            rmaxr0 = rmaxr0_new  # this is the final one
            rmaxr0_new = rmaxr0_new + drmaxr0
        # Check if solution converged
        if (not np.isnan(np.max(VV_ER11))) and ("rmerger0" in locals()):
            soln_converged = 1
        else:
            soln_converged = 0
            CkCd = CkCd + 0.1
            print("Adjusting CkCd to find convergence")

    # Calculate some things
    M0 = 0.5 * fcor * r0**2
    Mm = 0.5 * fcor * rmax**2 + rmax * Vmax
    MmM0 = Mm / M0

    # Finally: Interpolate to a grid
    ii_ER11 = np.argwhere((rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0))[:, 0]
    ii_E04 = np.argwhere((rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0))[:, 0]
    MMfracM0_temp = np.hstack((MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04]))
    rrfracr0_temp = np.hstack((rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04]))
    del ii_ER11
    del ii_E04

    # drfracr0 = .0001
    # rfracr0_min = 0        #[-] r=0
    # rfracr0_max = 1     #[-] r=r0
    # rrfracr0 = rfracr0_min:drfracr0:rfracr0_max #[] r/r0 vector
    # MMfracM0 = interp1(rrfracr0_temp,MMfracM0_temp,rrfracr0,'pchip',NaN)
    drfracrm = (
        0.01  # calculating VV at radii relative to rmax ensures no smoothing near rmax!
    )
    rfracrm_min = 0  # [-] r=0
    rfracrm_max = r0 / rmax  # [-] r=r0
    rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)  # [] r/r0 vector
    f = interp1d(rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm))
    MMfracMm = f(rrfracrm)

    rrfracr0 = rrfracrm * rmax / r0  # save this as output
    MMfracM0 = MMfracMm * Mm / M0

    # Calculate dimensional wind speed and radii
    # VV = (M0/r0)*((MMfracM0./rrfracr0)-rrfracr0)    #[ms-1]
    # rr = rrfracr0*r0     #[m]
    # rmerge = rmerger0*r0
    # Vmerge = (M0/r0)*((MmergeM0./rmerger0)-rmerger0)    #[ms-1]
    VV = (Mm / rmax) * (MMfracMm / rrfracrm) - 0.5 * fcor * rmax * rrfracrm  # [ms-1]
    rr = rrfracrm * rmax  # [m]

    # Make sure V=0 at r=0
    VV[rr == 0] = 0

    rmerge = rmerger0 * r0
    Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [ms-1]

    # Adjust profile in eye, if desired
    # if(eye_adj==1)
    # 	r_eye_outer = rmax
    # 	V_eye_outer = Vmax
    # 	[VV] = radprof_eyeadj(rr,VV,alpha_eye,r_eye_outer,V_eye_outer)

    #        sprintf('EYE ADJUSTMENT: eye alpha = #3.2f',alpha_eye)
    return rr, VV, rmerge, Vmerge, rmax


def ER11E04_nondim_rmaxinput(
    Vmax, rmax, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """ER11E04_nondim_rmaxinput

    Default values: Vmax=50, rmax=25*1000, fcor=5e-5, Cdvary=0, C_d=1.5e-3, w_cool=2/1000, CkCdvary=0, CkCd=1, eye_adj=0, alpha_eye=0.15

    Args:
        Vmax (float): Maximum azimuthally averaged wind speed [m/s]
        rmax (float): Radius at which Vmax occurs [m]
        fcor (float): Coriolis parameter [s-1]
        Cdvary (int): 1 if C_d varies with wind speed, 0 if constant
        C_d (float): Drag coefficient
        w_cool (float): Radiative subsidence rate. Cooling rate [K/s]


    Returns:
        rrfracr0 (np.ndarray): Radial coordinate [r/r0]
        MMfracM0 (np.ndarray): Mass coordinate [M/M0]
        rmerger0 (float): Radius at which mass coordinate is 1 [r0]
        Vmerge (float): Wind speed at rmerger0 [m/s]

    [rrfracr0,MMfracM0,rmerger0,Vmerge] = ER11E04_nondim_rmaxinput(Vmax,rmax,fcor,Cdvary,C_d,w_cool,CkCdvary,CkCd,eye_adj,alpha_eye)

    This function calculates the ER11+ER04 radial profile of the wind speed
    """
    # key function.
    # Initialization
    fcor = np.abs(fcor)
    if CkCdvary == 1:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst
    # 'Ck/Cd is capped at 1.9 and has been set to this value. If CkCdvary=1,
    # then Vmax is much greater than the range of data used to estimate
    # CkCd as a function of Vmax -- here be dragons!')
    CkCd = np.min((1.9, CkCd))
    # Step 1: Calculate ER11 M/Mm vs. r/rm
    # [~,~,rrfracrm_ER11,MMfracMm_ER11] = ER11_radprof_nondim(Vmax,rmax,fcor,CkCdvary,CkCd)
    drfracrm = 0.01
    if rmax > 100.0 * 1000:
        drfracrm = drfracrm / 10.0  # extra precision for large storm
    rfracrm_min = 0.0  # [-] start at r=0
    rfracrm_max = 50.0  # [-] extend out to many rmaxs
    rrfracrm_ER11 = np.arange(
        rfracrm_min, rfracrm_max + drfracrm, drfracrm
    )  # [] r/r0 vector
    rr_ER11 = rrfracrm_ER11 * rmax
    rmax_or_r0 = "rmax"
    soln_converged = 0
    count = 0
    while soln_converged == 0:
        count += 1
        VV_ER11, dummy = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)
        # Check if solution converged
        if not np.isnan(np.max(VV_ER11)):
            soln_converged = 1
        else:
            soln_converged = 0
            CkCd = CkCd + 0.1
            if rmax == 0.0:
                rmax = 25.0
            print("Adjusting CkCd to find convergence")
            if count >= 50:
                break
    if soln_converged == 1:
        Mm = 0.5 * fcor * rmax**2 + rmax * Vmax
        MMfracMm_ER11 = (rr_ER11 * VV_ER11 + 0.5 * fcor * rr_ER11**2) / Mm
        # Step 2: Converge rmaxr0 geometrically until ER11 M/M0 has tangent point with E04 M/M0
        # Break up interval into 3 points, take 2 between which intersection
        # vanishes, repeat til converges
        rmaxr0_min = 0.001
        rmaxr0_max = 0.75
        rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2  # first guess -- in the middle
        rmaxr0 = rmaxr0_new  # initialize
        drmaxr0 = rmaxr0_max - rmaxr0  # initialize
        drmaxr0_thresh = 0.000001
        iterN = 0
        while np.abs(drmaxr0) >= drmaxr0_thresh:
            iterN = iterN + 1
            # Calculate E04 M/M0 vs r/r0
            r0 = rmax / rmaxr0_new  # [m]
            Nr = 100000
            rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(
                r0, fcor, Cdvary, C_d, w_cool, Nr
            )
            # Convert ER11 to M/M0 vs. r/r0 space
            rrfracr0_ER11 = rrfracrm_ER11 * (rmaxr0_new)
            M0_E04 = 0.5 * fcor * r0**2
            MMfracM0_ER11 = MMfracMm_ER11 * (Mm / M0_E04)
            l1 = LineString(list(zip(rrfracr0_E04, MMfracM0_E04)))
            l2 = LineString(list(zip(rrfracr0_ER11, MMfracM0_ER11)))
            intersection = l1.intersection(l2)
            if (
                intersection.wkt == "GEOMETRYCOLLECTION EMPTY"
            ):  # no intersections r0 too large --> rmaxr0 too small
                drmaxr0 = np.abs(drmaxr0) / 2
            else:  # at least one intersection -- r0 too small --> rmaxr0 too large
                if intersection.wkt.split(" ")[0] == "POINT":
                    X0, Y0 = intersection.coords[0]
                elif intersection.wkt.split(" ")[0] == "MULTIPOINT":
                    X0, Y0 = intersection[0].coords[0]
                # at least one intersection -- rmaxr0 too large
                drmaxr0 = -np.abs(drmaxr0) / 2
                rmerger0 = np.mean(X0)
                MmergeM0 = np.mean(Y0)
            # update value of rmaxr0
            rmaxr0 = rmaxr0_new  # this is the final one
            rmaxr0_new = rmaxr0_new + drmaxr0
        if intersection.wkt != "GEOMETRYCOLLECTION EMPTY":
            # Calculate some things
            M0 = 0.5 * fcor * r0**2
            Mm = 0.5 * fcor * rmax**2 + rmax * Vmax
            MmM0 = Mm / M0

            # Finally: Interpolate to a grid
            # Finally: Interpolate to a grid
            ii_ER11 = np.argwhere(
                (rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0)
            )[:, 0]
            ii_E04 = np.argwhere(
                (rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0)
            )[:, 0]
            MMfracM0_temp = np.hstack((MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04]))
            rrfracr0_temp = np.hstack((rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04]))
            del ii_ER11
            del ii_E04
            rfracrm_min = 0  # [-] r=0
            rfracrm_max = r0 / rmax  # [-] r=r0
            rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)  # [] r/r0 vector
            f = interp1d(rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm))
            MMfracMm = f(rrfracrm)

            rrfracr0 = rrfracrm * rmax / r0  # save this as output
            MMfracM0 = MMfracMm * Mm / M0

            # Calculate dimensional wind speed and radii
            # VV = (M0/r0)*((MMfracM0./rrfracr0)-rrfracr0)    #[ms-1]
            # rr = rrfracr0*r0     #[m]
            # rmerge = rmerger0*r0
            # Vmerge = (M0/r0)*((MmergeM0./rmerger0)-rmerger0)    #[ms-1]
            VV = (Mm / rmax) * (
                MMfracMm / rrfracrm
            ) - 0.5 * fcor * rmax * rrfracrm  # [ms-1]
            rr = rrfracrm * rmax  # [m]

            # Make sure V=0 at r=0
            VV[rr == 0] = 0

            rmerge = rmerger0 * r0
            Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [ms-1]

            # Adjust profile in eye, if desired
            # if(eye_adj==1)
            # 	r_eye_outer = rmax
            # 	V_eye_outer = Vmax
            # 	[VV] = radprof_eyeadj(rr,VV,alpha_eye,r_eye_outer,V_eye_outer)
            #        sprintf('EYE ADJUSTMENT: eye alpha = #3.2f',alpha_eye)
        else:
            rr = rr_ER11
            VV = VV_ER11
            rmerge = np.float("nan")
            Vmerge = np.float("nan")
    else:
        rr = np.float("nan") * np.zeros(10)
        VV = np.float("nan") * np.zeros(10)
        r0 = np.float("nan")
        rmerge = np.float("nan")
        Vmerge = np.float("nan")
    return rr, VV, r0, rmerge, Vmerge


def ER11E04_nondim_rfitinput(
    Vmax, rfit, Vfit, fcor, Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye
):
    # Initialization
    fcor = np.abs(fcor)
    if CkCdvary == 1:
        CkCd_coefquad = 5.5041e-04
        CkCd_coeflin = -0.0259
        CkCd_coefcnst = 0.7627
        CkCd = CkCd_coefquad * Vmax**2 + CkCd_coeflin * Vmax + CkCd_coefcnst
    CkCd = np.min((1.9, CkCd))
    Mfit = rfit * Vfit + 0.5 * fcor * rfit**2
    soln_converged = 0
    while soln_converged == 0:

        rmaxrfit_min = 0.01
        rmaxrfit_max = 1.0
        rmaxrfit_new = (rmaxrfit_max + rmaxrfit_min) / 2  # first guess -- in the middle
        rmaxrfit = rmaxrfit_new  # initialize
        drmaxrfit = rmaxrfit_max - rmaxrfit  # initialize
        drmaxrfit_thresh = 0.0001
        # keep looping til changes in estimate are very small
        while np.abs(drmaxrfit) >= drmaxrfit_thresh:
            rmax = rmaxrfit_new * rfit
            # Step 1: Calculate ER11 M/Mm vs. r/rm
            # [~,~,rrfracrm_ER11,MMfracMm_ER11] = ER11_radprof_nondim(Vmax,rmax,fcor,CkCdvary,CkCd)
            drfracrm = 0.01
            if rmax > 100 * 1000:
                drfracrm = drfracrm / 10.0  # extra precision for large storm
            rfracrm_min = 0.0  # [-] start at r=0
            rfracrm_max = 50.0  # [-] extend out to many rmaxs
            rrfracrm_ER11 = np.arange(
                rfracrm_min, rfracrm_max + drfracrm, drfracrm
            )  # [] r/r0 vector
            rr_ER11 = rrfracrm_ER11 * rmax
            rmax_or_r0 = "rmax"
            VV_ER11, dummy = ER11_radprof(Vmax, rmax, rmax_or_r0, fcor, CkCd, rr_ER11)
            if not np.isnan(np.max(VV_ER11)):  # ER11_radprof converged
                Mm = rmax * Vmax + 0.5 * fcor * rmax**2
                MMfracMm_ER11 = (rr_ER11 * VV_ER11 + 0.5 * fcor * rr_ER11**2) / Mm
                rmaxr0_min = 0.01
                rmaxr0_max = 0.75
                # first guess -- in the middle
                rmaxr0_new = (rmaxr0_max + rmaxr0_min) / 2
                rmaxr0 = rmaxr0_new  # initialize
                drmaxr0 = rmaxr0_max - rmaxr0  # initialize
                drmaxr0_thresh = 0.000001
                iter = 0
                iterN = 0
                while np.abs(drmaxr0) >= drmaxr0_thresh:
                    iterN = iterN + 1
                    # Calculate E04 M/M0 vs r/r0
                    r0 = rmax / rmaxr0_new  # [m]
                    Nr = 100000
                    rrfracr0_E04, MMfracM0_E04 = E04_outerwind_r0input_nondim_MM0(
                        r0, fcor, Cdvary, C_d, w_cool, Nr
                    )
                    # Convert ER11 to M/M0 vs. r/r0 space
                    rrfracr0_ER11 = rrfracrm_ER11 * (rmaxr0_new)
                    M0_E04 = 0.5 * fcor * r0**2
                    MMfracM0_ER11 = MMfracMm_ER11 * (Mm / M0_E04)
                    l1 = LineString(list(zip(rrfracr0_E04, MMfracM0_E04)))
                    l2 = LineString(list(zip(rrfracr0_ER11, MMfracM0_ER11)))
                    intersection = l1.intersection(l2)

                    if (
                        intersection.wkt == "GEOMETRYCOLLECTION EMPTY"
                    ):  # no intersections r0 too large --> rmaxr0 too small
                        drmaxr0 = np.abs(drmaxr0) / 2
                    else:  # at least one intersection -- r0 too small --> rmaxr0 too large
                        if intersection.wkt.split(" ")[0] == "POINT":
                            X0, Y0 = intersection.coords[0]
                        elif intersection.wkt.split(" ")[0] == "MULTIPOINT":
                            X0, Y0 = intersection[0].coords[0]
                            # at least one intersection -- rmaxr0 too large
                        drmaxr0 = -np.abs(drmaxr0) / 2
                        rmerger0 = np.mean(X0)
                        MmergeM0 = np.mean(Y0)
                    # update value of rmaxr0
                    rmaxr0 = rmaxr0_new  # this is the final one
                    rmaxr0_new = rmaxr0_new + drmaxr0
                # Calculate some things
                M0 = 0.5 * fcor * r0**2
                Mm = 0.5 * fcor * rmax**2 + rmax * Vmax
                MmM0 = Mm / M0

                # Define merged solution
                ii_ER11 = np.argwhere(
                    (rrfracr0_ER11 < rmerger0) & (MMfracM0_ER11 < MmergeM0)
                )[:, 0]
                ii_E04 = np.argwhere(
                    (rrfracr0_E04 >= rmerger0) & (MMfracM0_E04 >= MmergeM0)
                )[:, 0]
                MMfracM0_temp = np.hstack(
                    (MMfracM0_ER11[ii_ER11], MMfracM0_E04[ii_E04])
                )
                rrfracr0_temp = np.hstack(
                    (rrfracr0_ER11[ii_ER11], rrfracr0_E04[ii_E04])
                )
                del ii_ER11
                del ii_E04

                # Check to see how close solution is to input value of
                # (rfitr0,MfitM0)
                rfitr0 = rfit / r0
                MfitM0 = Mfit / M0

                # 2020-06-23 fixed bug returning NaN (and wrong solution) if
                # rfit > current r0
                if rfitr0 <= 1:
                    f = interp1d(rrfracr0_temp, MMfracM0_temp)
                    # print(MMfracM0_temp)
                    MfitM0_temp = f(rfitr0)
                    MfitM0_err = MfitM0 - MfitM0_temp
                else:  # true rfit exceeds current r0, so doesnt exist in profile!
                    MfitM0_err = 1000000  # simply need smaller r0 -- set MfitM0_err to any positive number

                if MfitM0_err > 0:  # need smaller rmax (r0)
                    drmaxrfit = np.abs(drmaxrfit) / 2.0
                else:  # need larger rmax (r0)
                    drmaxrfit = -np.abs(drmaxrfit) / 2.0

            # ER11_radprof did not converge -- convergence fails for low CkCd
            # and high Ro = Vm/(f*rm)
            else:
                # Must reduce rmax (and thus reduce Ro)
                drmaxrfit = -abs(drmaxrfit) / 2

            # update value of rmaxrfit
            rmaxrfit = rmaxrfit_new  # this is the final one
            rmaxrfit_new = rmaxrfit + drmaxrfit
        # Check if solution converged
        if not np.isnan(np.max(VV_ER11)):
            soln_converged = 1
        else:
            soln_converged = 0
            CkCd = CkCd + 0.1
            print("Adjusting CkCd to find convergence")

    # Finally: Interpolate to a grid
    # drfracr0 = .0001
    # rfracr0_min = 0        #[-] r=0
    # rfracr0_max = 1     #[-] r=r0
    # rrfracr0 = rfracr0_min:drfracr0:rfracr0_max #[] r/r0 vector
    # MMfracM0 = interp1(rrfracr0_temp,MMfracM0_temp,rrfracr0,'pchip',NaN)
    drfracrm = (
        0.01  # calculating VV at radii relative to rmax ensures no smoothing near rmax!
    )
    rfracrm_min = 0.0  # [-] r=0
    rfracrm_max = r0 / rmax  # [-] r=r0
    rrfracrm = np.arange(rfracrm_min, rfracrm_max, drfracrm)  # [] r/r0 vector
    f = interp1d(rrfracr0_temp * (r0 / rmax), MMfracM0_temp * (M0 / Mm))
    MMfracMm = f(rrfracrm)
    rrfracr0 = rrfracrm * rmax / r0  # save this as output
    MMfracM0 = MMfracMm * Mm / M0

    # Calculate dimensional wind speed and radii
    # VV = (M0/r0)*((MMfracM0./rrfracr0)-rrfracr0)    #[ms-1]
    # rr = rrfracr0*r0     #[m]
    # rmerge = rmerger0*r0
    # Vmerge = (M0/r0)*((MmergeM0./rmerger0)-rmerger0)    #[ms-1]
    VV = (Mm / rmax) * (MMfracMm / rrfracrm) - 0.5 * fcor * rmax * rrfracrm  # [ms-1]
    rr = rrfracrm * rmax  # [m]

    # Make sure V=0 at r=0
    VV[rr == 0] = 0

    rmerge = rmerger0 * r0
    Vmerge = (M0 / r0) * ((MmergeM0 / rmerger0) - rmerger0)  # [ms-1]
    # Adjust profile in eye, if desired
    # if(eye_adj==1)
    #     r_eye_outer = rmax
    #     V_eye_outer = Vmax
    #     [VV] = radprof_eyeadj(rr,VV,alpha_eye,r_eye_outer,V_eye_outer)
    #     sprintf('EYE ADJUSTMENT: eye alpha = #3.2f',alpha_eye)
    return rr, VV, rmax, r0, rmerge, Vmerge
