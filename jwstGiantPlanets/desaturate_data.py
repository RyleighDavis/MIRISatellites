"""
Script to desaturate data. See desaturate_saturn.py for an example of using this.
"""

__version__ = '1.0.0'

import itertools
import math

import numpy as np
from astropy.io import fits
from scipy.ndimage import maximum_filter

from . import tools

EXPAND_WINDOW_SPECTRAL = 2
EXPAND_WINDOW_SPATIAL = 0
MAXIMUM_STEP = 1
MAXIMUM_STEP_TOTAL = 5
PARTIAL_SATURATION_FACTOR = 0.9
PARTIAL_SATURATION_SNR = 300

OUTLIER_FACTOR = 1.2
OUTLIER_THRESHOLD_LOW_SNR1 = 100
OUTLIER_FACTOR_LOW_SNR1 = 1.5
OUTLIER_THRESHOLD_LOW_SNR2 = 30
OUTLIER_FACTOR_LOW_SNR2 = 3


SKIP_GROUPS_MIRI_CHANNELS: dict[int, list[str]] = {
    1: ['1', '2'],
}


def replace_saturated(
    paths: list[str],
    path_out: str,
    keys: tuple[str, ...] = ('SCI', 'ERR'),
    sci_key: str = 'SCI',
    err_key: str = 'ERR',
    expand_window_spectral: int = EXPAND_WINDOW_SPECTRAL,
    expand_window_spatial: int = EXPAND_WINDOW_SPATIAL,
    maximum_step: int | None = None,
    maximum_step_total: int = MAXIMUM_STEP_TOTAL,
    partial_saturation_factor: float = PARTIAL_SATURATION_FACTOR,
    partial_saturation_snr: float = PARTIAL_SATURATION_SNR,
    partial_saturation_min_flux: float = 0,
    outlier_factor: float = OUTLIER_FACTOR,
    outlier_threshold_low_snr1: float = OUTLIER_THRESHOLD_LOW_SNR1,
    outlier_factor_low_snr1: float = OUTLIER_FACTOR_LOW_SNR1,
    outlier_threshold_low_snr2: float = OUTLIER_THRESHOLD_LOW_SNR2,
    outlier_factor_low_snr2: float = OUTLIER_FACTOR_LOW_SNR2,
    outlier_min_flux: float = 0,
    do_partial_saturation: bool = True,
    do_outlier_check: bool = True,
) -> None:
    """
    Create desaturated data cube using data with reduced groups.

    Args:
        paths: List of paths to data cubes. These should have a descending number of
            groups (e.g. first path has 5 groups, second has 4 groups, third has 3
            groups etc.).
        path_out: File path of desaturated data.
    """
    cube_arrays = {}
    dqs = []
    ngroups = []
    with fits.open(paths[0]) as hdul:
        for k in keys:
            cube_arrays[k] = [hdul[k].data]  # type: ignore
        dq = hdul['DQ'].data  # type: ignore
        dqs.append(dq)
        cube_arrays[sci_key][-1][dq != 0] = np.nan
        header = hdul['PRIMARY'].header  # type: ignore
        ngroups.append(header['NGROUPS'])
        instrument = header['INSTRUME']

        for p_reduced in paths[1:]:
            try:
                with fits.open(p_reduced) as hdul_reduced:
                    header_reduced = hdul_reduced['PRIMARY'].header  #  type: ignore
                    # Check that reduced group file is of same observation as main file
                    for k in [
                        'INSTRUME',
                        'DATE-BEG',
                        'PATT_NUM',
                        'CHANNEL',
                        'BAND',
                        'DETECTOR',
                        'FILTER',
                        'GRATING',
                    ]:
                        assert header.get(k) == header_reduced.get(k)
                    ng = header_reduced['NGROUPS']
                    if instrument == 'MIRI' and header_reduced[
                        'CHANNEL'
                    ] in SKIP_GROUPS_MIRI_CHANNELS.get(ng, []):
                        continue
                    ngroups.append(ng)
                    for k in keys:
                        cube_arrays[k].append(hdul_reduced[k].data)  #  type: ignore
                    dq = hdul_reduced['DQ'].data  # type: ignore
                    dqs.append(dq)
                    cube_arrays[sci_key][-1][dq != 0] = np.nan

            except FileNotFoundError:
                pass

        shape = cube_arrays[sci_key][0].shape
        indices_cube = np.full(shape, np.nan)
        flag_cube = np.full(shape, 0)
        for idx1 in range(shape[1]):
            for idx2 in range(shape[2]):
                spectra = [c[:, idx1, idx2] for c in cube_arrays[sci_key]]
                errors = [c[:, idx1, idx2] for c in cube_arrays[err_key]]
                sp, indices, flags = replace_saturated_spectra(
                    spectra,
                    errors,
                    expand_window_spectral=expand_window_spectral,
                    partial_saturation_factor=partial_saturation_factor,
                    partial_saturation_snr=partial_saturation_snr,
                    partial_saturation_min_flux=partial_saturation_min_flux,
                    outlier_factor=outlier_factor,
                    outlier_factor_low_snr1=outlier_factor_low_snr1,
                    outlier_threshold_low_snr1=outlier_threshold_low_snr1,
                    outlier_factor_low_snr2=outlier_factor_low_snr2,
                    outlier_threshold_low_snr2=outlier_threshold_low_snr2,
                    outlier_min_flux=outlier_min_flux,
                    do_partial_saturation=do_partial_saturation,
                    do_outlier_check=do_outlier_check,
                )
                indices_cube[:, idx1, idx2] = indices
                flag_cube[:, idx1, idx2] = flags
        if maximum_step is None:
            maximum_step = math.ceil(len(ngroups) / maximum_step_total)
        indices_cube = expand_spatial_windows(indices_cube, maximum_step=maximum_step)

        cubes = {k: np.full_like(c[0], np.nan) for k, c in cube_arrays.items()}
        ngroups_cube = np.full(shape, np.nan)
        for indices, group_idx in np.ndenumerate(indices_cube):
            if math.isnan(group_idx):
                continue
            group_idx = int(group_idx)
            ngroups_cube[indices] = ngroups[group_idx]
            for k, cube in cubes.items():
                cube[indices] = cube_arrays[k][group_idx][indices]

        for k in keys:
            hdul[k].data = cubes[k]  #  type: ignore
        hdul['DQ'].data[~np.isnan(cubes[sci_key])] = 0  #  type: ignore
        hdul['DQ'].data[np.isnan(cubes[sci_key])] = 1  #  type: ignore

        header = fits.Header()
        header.add_comment('Number of groups used when desaturating')
        hdu = fits.ImageHDU(
            data=np.nan_to_num(ngroups_cube).astype(np.uint8),
            header=header,
            name='NGROUPS',
        )
        hdul.append(hdu)

        header = fits.Header()
        header.add_comment('Type of bad data identified when desaturating')
        header.add_comment('0: Good data')
        header.add_comment(
            '1: Bad data identified by standard pipeline (e.g. saturated)'
        )
        header.add_comment('2: Partially saturated')
        header.add_comment('3: Outlier (e.g. cosmic ray)')
        hdu = fits.ImageHDU(
            data=flag_cube.astype(np.uint8),
            header=header,
            name='DESAT_TYPE',
        )
        hdul.append(hdu)

        tools.add_header_reduction_note(hdul, 'Desaturated')
        header = hdul['PRIMARY'].header  #  type: ignore
        header['NGROUPS'] = 'Desaturated'
        header['HIERARCH DESAT VERSION'] = (__version__, 'Software version')
        header['HIERARCH DESAT NFILES'] = (
            len(ngroups),
            'Number of files used in desaturation',
        )
        if len(ngroups) >= 10:
            groups_str = f'{ngroups[0]},{ngroups[1]},...,{ngroups[-2],ngroups[-1]}'
        else:
            groups_str = ','.join(str(n) for n in ngroups)
        header['HIERARCH DESAT GROUPS'] = (
            groups_str,
            'Group options in desaturation',
        )
        header['HIERARCH DESAT PARTIAL_SAT_FACTOR'] = (
            partial_saturation_factor,
            'Partial saturation factor',
        )
        header['HIERARCH DESAT PARTIAL_SAT_SNR'] = (
            partial_saturation_snr,
            'Partial saturation min. SNR',
        )
        header['HIERARCH DESAT PARTIAL_SAT_MIN_FLUX'] = (
            partial_saturation_min_flux,
            'Partial saturation min. flux',
        )
        header['HIERARCH DESAT OUTLIER_FACTOR'] = (outlier_factor,)
        header['HIERARCH DESAT OUTLIER_FACTOR_LOW_SNR1'] = (outlier_factor_low_snr1,)
        header['HIERARCH DESAT OUTLIER_FACTOR_LOW_SNR2'] = (outlier_factor_low_snr2,)
        header['HIERARCH DESAT OUTLIER_THRESHOLD_LOW_SNR1'] = (
            outlier_threshold_low_snr1,
        )
        header['HIERARCH DESAT OUTLIER_THRESHOLD_LOW_SNR2'] = (
            outlier_threshold_low_snr2,
        )
        header['HIERARCH DESAT OUTLIER_MIN_FLUX'] = (outlier_min_flux,)
        header['HIERARCH DESAT EXPAND_SPECTRAL'] = (
            expand_window_spectral,
            'Window expansion size, spectral dimension',
        )
        header['HIERARCH DESAT EXPAND_SPATIAL'] = (
            expand_window_spatial,
            'Window expansion size, spatial dimension',
        )
        header['HIERARCH DESAT MAX_STEP'] = (
            maximum_step,
            'Max group step (spectral & spatial dimensions)',
        )
        header['HIERARCH DESAT MAX_STEP_TOTAL'] = (
            maximum_step_total,
            'Max total group step',
        )
        header['HIERARCH DESAT DO_PARTIAL_SATURATION'] = (
            do_partial_saturation,
            'Check for partial saturation',
        )
        header['HIERARCH DESAT DO_OUTLIER_CHECK'] = (
            do_outlier_check,
            'Check for outliers (e.g. cosmic rays)',
        )
        tools.check_path(path_out)
        hdul.writeto(path_out, overwrite=True)


def replace_saturated_spectra(
    spectra: list[np.ndarray],
    errors: list[np.ndarray],
    expand_window_spectral: int = EXPAND_WINDOW_SPECTRAL,
    partial_saturation_factor: float = PARTIAL_SATURATION_FACTOR,
    partial_saturation_snr: float = PARTIAL_SATURATION_SNR,
    partial_saturation_min_flux: float = 0,
    outlier_factor: float = OUTLIER_FACTOR,
    outlier_factor_low_snr1: float = OUTLIER_FACTOR_LOW_SNR1,
    outlier_factor_low_snr2: float = OUTLIER_FACTOR_LOW_SNR2,
    outlier_threshold_low_snr1: float = OUTLIER_THRESHOLD_LOW_SNR1,
    outlier_threshold_low_snr2: float = OUTLIER_THRESHOLD_LOW_SNR2,
    outlier_min_flux: float = 0,
    do_partial_saturation: bool = True,
    do_outlier_check: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = [saturated_to_nan(s) for s in spectra]
    sp = np.full(spectra[0].shape, np.nan)
    indices = np.full(sp.shape, np.nan)
    sp_min_groups = spectra[-1]
    desat_flags = np.full(sp.shape, 0)
    for sp_idx, sp_reduced in enumerate(spectra):
        snr = sp_reduced / errors[sp_idx]

        # Flag general bad values in data
        bad_nan = np.isnan(sp)

        # Flag partially saturated values where the new spectrum doesn't look like a
        # cosmic ray (i.e. it's not an outlier), and has a SNR above the threshold
        if do_partial_saturation:
            bad_sat = (
                (indices == sp_idx - 1)
                & (sp > partial_saturation_min_flux)
                & (sp_reduced > partial_saturation_min_flux)
                & (snr > partial_saturation_snr)
                & (sp < sp_reduced * partial_saturation_factor)
                & (sp_reduced < sp_min_groups * outlier_factor)
            )
        else:
            bad_sat = np.zeros(sp.shape, dtype=bool)

        # Flag values which seem like a cosmic ray (i.e. +ve outlier)
        if do_outlier_check:
            bad_cosmic_ray = (
                (indices == sp_idx - 1)
                & (sp > outlier_min_flux)
                & (sp_reduced > outlier_min_flux)
                & (
                    (
                        (sp > sp_reduced * outlier_factor)
                        & (snr > outlier_threshold_low_snr1)
                    )
                    | (
                        (sp > sp_reduced * outlier_factor_low_snr1)
                        & (snr > outlier_threshold_low_snr2)
                    )
                    | (sp > sp_reduced * outlier_factor_low_snr2)
                )
            )
        else:
            bad_cosmic_ray = np.zeros(sp.shape, dtype=bool)

        bad_combined = bad_nan | bad_sat | bad_cosmic_ray

        if sp_idx > 0:
            # On first pass, everything is replaced (as sp is initialised as all NaN)
            desat_flags[bad_cosmic_ray] = 3
            desat_flags[bad_sat] = 2
            desat_flags[bad_nan] = 1

        bad_indices = get_indices(bad_combined)
        for a, b in bad_indices:
            a = max(a - expand_window_spectral, 0)
            b = b + expand_window_spectral
            if all(np.isnan(sp_reduced[a : b + 1])):
                sp[a : b + 1] = np.nan
                continue
            sp[a : b + 1] = sp_reduced[a : b + 1]
            indices[a : b + 1] = sp_idx
    return sp, indices, desat_flags


def expand_spatial_windows(
    indices_cube: np.ndarray,
    expand_window_spatial: int = EXPAND_WINDOW_SPATIAL,
    maximum_step: int = MAXIMUM_STEP,
) -> np.ndarray:
    nans = np.isnan(indices_cube)
    cube = indices_cube.copy()
    cube[nans] = 0
    if expand_window_spatial:
        size = 1 + 2 * expand_window_spatial
        for idx, img in enumerate(cube):
            cube[idx] = maximum_filter(img, size=size)

    if maximum_step:
        for _ in range(int(np.max(cube))):
            step = maximum_filter(cube, 3) - maximum_step  #  type: ignore
            cube = np.maximum(cube, step)

    cube[nans] = np.nan
    return cube


def saturated_to_nan(sp: np.ndarray) -> np.ndarray:
    sp = sp.copy()
    sp[sp == 0] = np.nan
    return sp


def get_indices(bad_flags: np.ndarray):
    # https://stackoverflow.com/questions/57611413/python-return-start-stop-indices-of-values
    last_index = 0
    out = []
    for bad, g in itertools.groupby(enumerate(bad_flags), lambda k: k[1]):
        l = [*g]
        if bad:
            out.append([last_index, l[-1][0]])
        last_index += len(l)
    return out
