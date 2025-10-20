### This is where I created the stpsf (formerly WebbPSF) psf's for each wavelength in MIRI IFU, only need 
### to run this once and will use the pickle file created for all of the galilean satellite MIRI observations

import stpsf
miri = stpsf.MIRI()
miri.mode = 'IFU'

import numpy as np
import pickle
from scipy.ndimage import shift
from tqdm import tqdm
import planetmapper

def extract_odd_psf_kernel(psf_data, kernel_size=None, subpixel_shift=0.5, threshold=1e-6):
    """
    Extract an odd-sized PSF kernel with proper centering
    
    Parameters:
    -----------
    psf_data : 2D or 3D array
        PSF data (if 3D, shape is (nwave, ny, nx))
    kernel_size : int or None
        Desired kernel size (must be odd). If None, auto-determined
    subpixel_shift (0.5) : none or shift value
        Whether to apply 0.5 pixel shift to center the PSF. Only if
        psf is centered on integer pixels
    threshold: float
        Threshold to use if no kernel_size is provided. Where PSF drops to
        this fraction of its peak value.
        
    Returns:
    --------
    shifted_psf : array
        PSF kernel(s) with odd dimensions and proper centering
    """
    def _extract_single_odd_psf(psf_2d, kernel_size, subpixel_shift, threshold=1e-6):
        """Extract single odd-sized PSF with centering"""
        
        ny, nx = psf_2d.shape
        
        # Apply subpixel shift to center the PSF properly
        if subpixel_shift is not None:
            # Shift by 0.5 pixels in each direction to center between pixels
            shifted_psf = shift(psf_2d, shift=(subpixel_shift, subpixel_shift), order=3, mode='constant', cval=0.0)
        else:
            shifted_psf = psf_2d.copy()
        
        # Find the center
        cy, cx = ny // 2, nx // 2
        
        # Determine kernel size if not provided
        if kernel_size is None:
            # Auto-determine based on where PSF drops to some fraction of peak
            peak_val = np.max(shifted_psf)
            threshold = peak_val * threshold  # PSF drops to 1e-6 of peak
            
            # Find extent where PSF is above threshold
            above_thresh = shifted_psf > threshold
            y_indices, x_indices = np.where(above_thresh)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_extent = np.max(y_indices) - np.min(y_indices)
                x_extent = np.max(x_indices) - np.min(x_indices)
                max_extent = max(y_extent, x_extent)
                
                # Add some padding and ensure odd
                kernel_size = max_extent + 20  # Add padding
                if kernel_size % 2 == 0:
                    kernel_size += 1
            else:
                # Fallback
                print('Automatic kernel size could not be determined. Falling back to entire image minus last row/column.')
                kernel_size = min(ny, nx)
                if kernel_size % 2 == 0:
                    kernel_size -= 1
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Calculate extraction bounds
        half_size = kernel_size // 2
        
        # Get bounds
        y_start = max(0, cy - half_size)
        y_end = min(ny, cy + half_size + 1)
        x_start = max(0, cx - half_size)
        x_end = min(nx, cx + half_size + 1)
        
        # Extract the kernel
        extracted = shifted_psf[y_start:y_end, x_start:x_end]
        
        # If extraction is smaller than desired size due to edge effects, pad with zeros
        if extracted.shape != (kernel_size, kernel_size):
            #print('Extracted region is smaller than kernel size. Padding edges with zeros.')
            padded = np.zeros((kernel_size, kernel_size))
            
            # Calculate where to place the extracted region in the padded array
            pad_y_start = (kernel_size - extracted.shape[0]) // 2
            pad_x_start = (kernel_size - extracted.shape[1]) // 2
            
            padded[pad_y_start:pad_y_start + extracted.shape[0],
                pad_x_start:pad_x_start + extracted.shape[1]] = extracted
            
            extracted = padded
        
        return extracted
    
    if psf_data.ndim == 2:
        # Single PSF image
        return _extract_single_odd_psf(psf_data, kernel_size, subpixel_shift, threshold)
    
    elif psf_data.ndim == 3:
        # PSF cube - process each wavelength
        nwave = psf_data.shape[0]
        
        # Determine kernel size from last PSF (should be longest wavelength) if not provided
        if kernel_size is None:
            test_kernel = _extract_single_odd_psf(psf_data[-1], None, subpixel_shift, threshold)
            kernel_size = test_kernel.shape[0]
        
        # Process all wavelengths
        shifted_cube = np.zeros((nwave, kernel_size, kernel_size))
        
        for i in range(nwave):
            shifted_cube[i] = _extract_single_odd_psf(
                psf_data[i], kernel_size, subpixel_shift
            )
        
        return shifted_cube
    
    else:
        raise ValueError("PSF data must be 2D (nx, ny) or 3D array (nwaves,nx,ny).")



if __name__=='__main__':
    ### Create and store dictionary of wavelength-dependent PSFs for each channel and band ###
    # psfs = {}
    # datapath = "/home/rdavis/DataAnalysis/data/jwst/miri/callisto/reduced/leading70W/stage3_desaturated"
    # psf_filenm = f"/home/rdavis/DataAnalysis/pipelines/MIRISatellites/jwstGiantPlanets/stpsfs.pkl"
    # dith = 'd1'
    # miri.detector = 'MIRIFUSHORT'
    # for ch in ['1','2']:
    #     for band in ['SHORT', 'MEDIUM', 'LONG']:
    #         obsfilenm = f"{datapath}/{dith}_bg_fringe_fringe1d_cropped/Level3_ch{ch}-{band.lower()}_s3d_fixednav_cropped.fits"
    #         body = planetmapper.Observation(obsfilenm)
    #         waves = body.get_wavelengths_from_header()*u.m # in m
    #         print(f"Getting psfs for channel {ch} {band}: [{np.round(waves.to(u.um).min(),2)}:{np.round(waves.to(u.um).max(),2)}]")
    #         p = miri.calc_datacube(waves.to(u.m).value, progressbar=True, fov_arcsec=2,  oversample=5)
    #         psfs[f"({ch},{band})"] = p

    #         with open(psf_filenm, 'wb') as f:
    #             pickle.dump(psfs, f)
    # #update ch 3 and 4 detector
    # miri.detector = 'MIRIFULONG'
    # for ch in ['3','4']:
    #     for band in ['SHORT', 'MEDIUM', 'LONG']:
    #         obsfilenm = f"{datapath}/{dith}_bg_fringe_fringe1d_cropped/Level3_ch{ch}-{band.lower()}_s3d_fixednav_cropped.fits"
    #         body = planetmapper.Observation(obsfilenm)
    #         waves = body.get_wavelengths_from_header()*u.m # in m
    #         print(f"Getting psfs for channel {ch} {band}: [{np.round(waves.to(u.um).min(),2)}:{np.round(waves.to(u.um).max(),2)}]")
    #         p = miri.calc_datacube(waves.to(u.m).value, progressbar=True, fov_arcsec=2,  oversample=5)
    #         psfs[f"({ch},{band})"] = p

    #         with open(psf_filenm, 'wb') as f:
    #             pickle.dump(psfs, f)

    ### Load previously created PSF object because the above code is really slow  ###
    psf_filenm = f"/home/rdavis/DataAnalysis/pipelines/MIRISatellites/jwstGiantPlanets/stpsfs.pkl"
    with open(psf_filenm, 'rb') as f:
        psfs = pickle.load(f)

    ### Now, extract odd-sized kernels for use in colvolution ###
    kernels = {}
    for ch in ['1','2','3','4']:
        for band in ['SHORT', 'MEDIUM', 'LONG']:
            print(f'Extracting odd-sized PSF kernels for channel {ch} {band}...')
            # Extract odd-sized kernel with proper centering
            # Here, I am using a subpixel shift of 0.5 to center the PSF between pixels
            # and letting the function determine the kernel size automatically based on threshold
            # If you want to specify a fixed kernel size, set kernel_size to an odd integer
            # e.g., kernel_size=101
            psf_cube_data = extract_odd_psf_kernel(psfs[f'({ch},{band})'][0].data, subpixel_shift=0.5, threshold=0.01)
            kernels[f'({ch},{band})'] = psf_cube_data
    kernel_filenm = f"/home/rdavis/DataAnalysis/pipelines/MIRISatellites/jwstGiantPlanets/miri_psf_kernels.pkl"
    with open(kernel_filenm, 'wb') as f:
        pickle.dump(kernels, f)