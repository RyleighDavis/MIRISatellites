### code to model the solar reflectance for subtraction from MIRI observations ###

import numpy as np
import astropy.io.fits as apfits
import astropy.units as u
from astropy.constants import c, h, k_B
from scipy.interpolate import interp1d
import planetmapper
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from tqdm import tqdm


jpl_ids = {'EUROPA': '502',
           'GANYMEDE': '503',
           'CALLISTO': '504',}
albedos = {'EUROPA': {'lead':0.65 ,
                      'trail': 0.55},
           'GANYMEDE': {'lead': 0.43,
                      'trail': 0.37},
           'CALLISTO': {'lead': 0.2,
                      'trail': 0.22}
            }

### Estimate solar reflected component to subtract ###
def lambert_phase_function(local_phase_angle_rad):
    """
    Lambert sphere phase function.
    
    Parameters:
    -----------
    local_phase_angle_rad : float or array
        Local phase angle in radians
        
    Returns:
    --------
    phase_function : float or array
        Lambert phase function value: Φ(α) = (sin(α) + (π - α) * cos(α)) / π
    """
    return (np.sin(local_phase_angle_rad) + 
            (np.pi - local_phase_angle_rad) * np.cos(local_phase_angle_rad)) / np.pi

def calculate_reflected_solar_flux(wavelength, pixel_coords, satellite_radius_arcsec, 
                                 sun_satellite_distance, satellite_observer_distance,
                                 pixel_size_arcsec, albedo, phase_angle,
                                 solar_spectrum=None, scattering_function=None):
    """
    Calculate the absolute flux of the reflected solar component for any Galilean satellite IFU observation.
    
    Parameters:
    -----------
    wavelength : array_like
        Wavelength array in microns
    pixel_coords : array_like, shape (ny, nx)
        2D array where each pixel contains its angular distance from disk center in arcseconds.
        Values should be NaN for pixels outside the satellite disk.
    satellite_radius_arcsec : float
        Satellite's apparent radius in arcseconds during the observation
    sun_satellite_distance : float
        Sun-satellite distance in AU
    satellite_observer_distance : float
        Satellite-Observer distance in AU
    pixel_size_arcsec : float
        Angular size of each IFU pixel in arcseconds
    albedo : float
        Constant geometric albedo value for all pixels
    phase_angle : float
        Phase angle in degrees (Sun-satellite-Observer angle)
    solar_spectrum : array_like, optional
        Solar flux spectrum at 1 AU in W m-2 μm-1. If None, uses blackbody at 5778K
    scattering_function : callable, optional
        Phase function for scattering. Takes local phase angle in radians as input.
        If None, uses Lambert sphere phase function.
        
    Returns:
    --------
    reflected_flux : array_like, shape (ny, nx, len(wavelength))
        Reflected solar flux for each pixel in W m-2 μm-1
    disk_integrated_flux : array_like, shape (len(wavelength),)
        Disk-integrated reflected flux in W m-2 μm-1
    """
    
    # Convert units
    wavelength = np.asarray(wavelength)  # microns
    pixel_coords = np.asarray(pixel_coords)  # arcseconds, shape (ny, nx)
    phase_angle_rad = np.radians(phase_angle)
    
    # Solar spectrum at 1 AU
    if solar_spectrum is None:
        # Use solar blackbody at 5778 K
        T_sun = 5778  # K
        solar_flux_1au = planck_function(wavelength, T_sun) * np.pi  # Convert to flux
        # Scale to realistic solar flux levels (~1400 W/m2 integrated)
        solar_flux_1au *= 2.04e-5  # Empirical scaling factor
    else:
        # Interpolate provided solar spectrum to our wavelength grid
        solar_interp = interp1d(solar_spectrum[:, 0], solar_spectrum[:, 1], 
                               bounds_error=False, fill_value=0.0)
        solar_flux_1au = solar_interp(wavelength)
    
    # Scale solar flux to satellite's distance
    solar_flux_satellite = solar_flux_1au / (sun_satellite_distance**2)
    
    # Calculate pixel solid angle (in steradians)
    pixel_area_sr = (pixel_size_arcsec * u.arcsec).to(u.rad).value**2
    
    # Get array dimensions
    ny, nx = pixel_coords.shape
    n_wavelengths = len(wavelength)
    
    # Initialize output array
    reflected_flux = np.zeros((n_wavelengths, ny, nx))
    
    # Set default scattering function if none provided
    if scattering_function is None:
        scattering_function = lambert_phase_function
    
    # Find valid pixels (not NaN and within disk)
    valid_pixels = ~np.isnan(pixel_coords)
    
    # Calculate reflected flux for each valid pixel
    for i in range(ny):
        for j in range(nx):
            if not valid_pixels[i, j]:
                continue
                
            # Distance from satellite center in arcseconds
            r_arcsec = pixel_coords[i, j]
            
            # Skip pixels outside satellite's disk (should not happen with NaN masking)
            if r_arcsec > satellite_radius_arcsec:
                continue
                
            # Calculate pixel position on the sphere
            # Convert from center-distance to x,y coordinates
            # Assuming i corresponds to y-direction, j to x-direction
            x_arcsec = (j - nx//2) * pixel_size_arcsec
            y_arcsec = (i - ny//2) * pixel_size_arcsec
            
            # Calculate local incidence and emission angles
            # Foreshortening factor (cosine of emission angle to observer)
            mu = np.sqrt(1 - (r_arcsec/satellite_radius_arcsec)**2) if r_arcsec < satellite_radius_arcsec else 0
            
            # Calculate local phase angle for this pixel
            # This accounts for the surface normal at this location on the sphere
            
            # Surface normal direction (pointing toward observer at disk center)
            # At disk center: normal = (0, 0, 1)
            # At edge: normal has x,y components
            if r_arcsec < satellite_radius_arcsec:
                # Normalized surface normal vector
                normal_x = (x_arcsec / satellite_radius_arcsec) * np.sqrt(1 - mu**2)
                normal_y = (y_arcsec / satellite_radius_arcsec) * np.sqrt(1 - mu**2)
                normal_z = mu
                
                # Sun direction vector (approximation: sun illuminates from the side at global phase angle)
                # This is a simplified geometry - for exact calculation would need full 3D positions
                sun_x = -np.sin(phase_angle_rad)  # Sun offset in x-direction
                sun_y = 0                         # Assume sun is in x-z plane
                sun_z = -np.cos(phase_angle_rad)  # Sun illumination component
                
                # Observer direction (toward viewer, z-direction)
                obs_x, obs_y, obs_z = 0, 0, 1
                
                # Calculate local incidence angle (sun-surface angle)
                cos_incidence = -(normal_x * sun_x + normal_y * sun_y + normal_z * sun_z)
                cos_incidence = max(0, cos_incidence)  # No negative illumination
                
                # Calculate local emission angle (surface-observer angle)  
                cos_emission = normal_x * obs_x + normal_y * obs_y + normal_z * obs_z
                cos_emission = max(0, cos_emission)
                
                # Calculate local phase angle between incident and reflected rays
                # cos(phase) = cos(incidence) * cos(emission) + sin(incidence) * sin(emission) * cos(azimuth_diff)
                # For simplified geometry, approximate local phase angle
                if cos_incidence > 0 and cos_emission > 0:
                    # Dot product of sun and observer directions in surface frame
                    sun_dot_obs = sun_x * obs_x + sun_y * obs_y + sun_z * obs_z
                    # Local phase angle accounting for surface curvature
                    cos_local_phase = (cos_incidence * cos_emission + 
                                     np.sqrt(1 - cos_incidence**2) * np.sqrt(1 - cos_emission**2) * sun_dot_obs)
                    cos_local_phase = np.clip(cos_local_phase, -1, 1)
                    local_phase_angle = np.arccos(cos_local_phase)
                else:
                    local_phase_angle = phase_angle_rad  # Fallback
                    cos_incidence = 0
                    cos_emission = 0
            else:
                continue
            
            if cos_incidence <= 0 or cos_emission <= 0:  # Pixel is in shadow or not visible
                continue
            
            # Calculate phase function for this pixel's local phase angle
            local_phase_function = scattering_function(local_phase_angle)
            
            # Calculate reflected flux for this pixel using local geometry
            # I/F = (albedo * phase_function * cos_incidence) / (4 * cos_emission)
            i_over_f = (albedo * local_phase_function * cos_incidence) / (4 * cos_emission)
            
            # Convert to absolute flux
            # Flux = (I/F) * (solar_flux / π) * (pixel_solid_angle)
            reflected_flux[:, i, j] = (i_over_f * solar_flux_satellite * pixel_area_sr) / np.pi
    
    # Calculate disk-integrated flux (sum over valid pixels)
    disk_integrated_flux = np.nansum(reflected_flux, axis=(1, 2))
    
    # Convert from W m-2 μm-1 to Jy
    # Conversion: F_ν [Jy] = F_λ [W m-2 μm-1] * λ^2 [μm^2] / c [m s-1] * 10^26
    # where c = 2.998e8 m/s and 10^26 converts W m-2 Hz-1 to Jy
    c_mks = 2.998e8  # m/s
    conversion_factor = (wavelength**2) / c_mks * 1e26  # Convert to Jy
    
    # Apply conversion to both outputs
    reflected_flux_jy = reflected_flux * conversion_factor[:,np.newaxis, np.newaxis]
    disk_integrated_flux_jy = disk_integrated_flux * conversion_factor
    
    return reflected_flux_jy, disk_integrated_flux_jy

def planck_function(wavelength_microns, temperature):
    """
    Calculate Planck function in units of W m-2 μm-1 sr-1
    
    Parameters:
    -----------
    wavelength_microns : array_like
        Wavelength in microns
    temperature : float
        Temperature in Kelvin
        
    Returns:
    --------
    planck : array_like
        Planck function in W m-2 μm-1 sr-1
    """
    wavelength_m = wavelength_microns * 1e-6
    
    # Planck function
    c1 = 2 * h.value * c.value**2
    c2 = h.value * c.value / k_B.value
    
    planck = c1 / (wavelength_m**5) / (np.exp(c2 / (wavelength_m * temperature)) - 1)
    
    # Convert from W m-2 m-1 sr-1 to W m-2 μm-1 sr-1
    planck *= 1e-6
    
    return planck

def create_pixel_coords_2d(ny, nx, x0, y0, pixel_scale_arcsec, satellite_radius_arcsec):
    """
    Create a 2D pixel coordinate array where each pixel contains its angular distance from disk center.
    
    Parameters:
    -----------
    ny, nx : int
        Spatial dimensions of the IFU
    x0, y0 : float
        Disk center coordinates in pixels (can be fractional)
    pixel_scale_arcsec : float
        Angular size of each pixel in arcseconds
    satellite_radius_arcsec : float
        Satellite radius in arcseconds
        
    Returns:
    --------
    pixel_coords : array, shape (ny, nx)
        2D array where each pixel contains its angular distance from disk center.
        Pixels outside the disk are set to NaN.
    """
    
    # Create coordinate grids
    j_grid, i_grid = np.meshgrid(range(nx), range(ny))  # Note: j=x, i=y
    
    # Calculate offset from disk center in pixels
    dx_pixels = j_grid - x0
    dy_pixels = i_grid - y0
    
    # Convert to angular distance from center
    dx_arcsec = dx_pixels * pixel_scale_arcsec
    dy_arcsec = dy_pixels * pixel_scale_arcsec
    
    # Calculate radial distance from center
    r_arcsec = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
    
    # Set pixels outside the disk to NaN
    pixel_coords = r_arcsec.copy()
    pixel_coords[r_arcsec > satellite_radius_arcsec] = np.nan
    
    return pixel_coords


def add_solar_model_to_obsfiles(datapath, inertias=[15,30,50,100,200]):
    ### Caclculate reflected solar spectrum ###
    solspec = np.loadtxt('/net/eris/data1/data/JWST/miri/miri_solarmodel.txt', skiprows=17)
    # absolute magnitude in units of angstrom, and erg/s/cm2/A
    # # Convert solar spectrum to units of Jy and microns (to match JWST data)
    # sol[:,1] = 3.33564095E+04*sol[:,1]*sol[:,0]**2
    # sol[:,0] = sol[:,0]/10000
    solspec[:,0] = solspec[:,0] * 1e-4 # A → microns
    # Combined conversion factor:
        # erg/s/cm²/Å → W/m²/μm = (1e-7 W/erg/s) × (1e4 m²/cm²) × (1e4 μm/Å)
        # = 1e-7 × 1e4 × 1e4 = 1e1 = 10
    solspec[:,1] = solspec[:,1] * 10

    total_iterations = len(datapath.keys()) * len(['d1', 'd2', 'd3', 'd4']) * len(['1', '2', '3', '4']) * len(['SHORT', 'MEDIUM', 'LONG']) * len(inertias)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for hem in datapath.keys():
                for dith in ['d1', 'd2', 'd3', 'd4']:
                    for ch in ['1', '2', '3', '4']:
                        for band in ['SHORT', 'MEDIUM', 'LONG']:
                            for inertia in inertias:
                                obsfilenm = f"{datapath[hem]}/{dith}/TI_{inertia}/Level3_ch{ch}-{band.lower()}_s3d_psf.fits"
                                body = planetmapper.Observation(obsfilenm)

                                pixel_coords = create_pixel_coords_2d(body.data.shape[1], body.data.shape[2], 
                                                                    body.get_x0(), body.get_y0(), # disk cen pixel coords
                                                                    body.get_plate_scale_arcsec(), # plate scale in arcsec
                                                                    body.get_r0()*body.get_plate_scale_arcsec() # radius in arcsec
                                                                    )
                                eph = Horizons(id=jpl_ids[body.target],  
                                            location='@jwst',  # Observer at JWST
                                            epochs=Time(body.utc).jd).ephemerides()

                                solflux, diskint_solflux = calculate_reflected_solar_flux((body.get_wavelengths_from_header()*u.m).to(u.um).value, # wavelength in microns 
                                                                                        pixel_coords, # pixel distance from disk center in arcseconds, 2D array
                                                                                        body.get_r0()*body.get_plate_scale_arcsec(), # radius in arcsec
                                                                                        eph['r'].value.data[0], # sun_satellite_distance
                                                                                        eph['delta'].value.data[0], #satellite_observer_distance,
                                                                                        body.get_plate_scale_arcsec(),# pixel_size_arcsec
                                                                                        albedos[body.target][hem], #albedo
                                                                                        eph['alpha'].value.data[0], #phase_angle
                                                                                        solar_spectrum=solspec, 
                                                                                        scattering_function=None)

                                ### Add to observation fits file ###
                                obs_hdul = apfits.open(obsfilenm)
                                # Add solar reflected cube
                                newhdu = apfits.ImageHDU(data=solflux, name="REFLECTED_SOLAR_FLUX")
                                newhdu.header['BUNIT'] = 'Jy' 
                                newhdu.header['ALBEDO'] = albedos[body.target][hem]
                                newhdu.header['SUNSAT'] = eph['r'].value.data[0] # sun satellite distance
                                newhdu.header['SATOBS'] = eph['delta'].value.data[0] # satellite observer distance
                                newhdu.header['PHASEANG'] = eph['alpha'].value.data[0] # phase angle
                                newhdu.header.add_comment('Solar reflectance model in Jy per pixel')
                                obs_hdul.append(newhdu)
                            
                                cols = [apfits.Column(name='WAVELENGTH', format='D', unit='um', array=(body.get_wavelengths_from_header()*u.m).to(u.um).value), 
                                        apfits.Column(name='REFLECTED_SOLAR', format='D', unit='Jy', array=diskint_solflux),
                                        ]
                                newhdu = apfits.BinTableHDU.from_columns(cols, name='DISKINT_SOLAR_FLUX') 
                                newhdu.header.add_comment(f'Disk integrated solar reflectance model in Jy')
                                obs_hdul.append(newhdu)

                                # re-write to disk with solar reflectance model added
                                obs_hdul.writeto(obsfilenm, overwrite=True)
                                pbar.update(1)