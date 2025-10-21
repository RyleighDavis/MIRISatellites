__version__ = '1.0.0'

import os

import numpy as np
from astropy.io import fits

from . import tools


def detect_saturation_end_group(path: str) -> int:
    """
    Detect the group number where saturation ends for a FITS file.
    
    Args:
        path: Path to the FITS file
        
    Returns:
        Group number (1-indexed) where saturation effectively ends.
        Returns the full number of groups if no significant saturation is detected.
    """
    with fits.open(path) as hdul:
        data = hdul['SCI'].data  # Shape: (integrations, groups, y, x)
        dq = hdul['DQ'].data if 'DQ' in hdul else None  # DQ flags
        ngroups_full = hdul[0].header['NGROUPS']  # type: ignore
        
        # For each group, count saturated/bad pixels across all integrations and spatial pixels
        saturated_counts = []
        
        # DO_NOT_USE flag value in JWST DQ arrays
        DO_NOT_USE = 1  # This is the standard JWST DQ flag for DO_NOT_USE
        
        for group_idx in range(ngroups_full):
            group_data = data[:, group_idx, :, :]
            
            # Count pixels flagged as DO_NOT_USE (which includes saturation)
            saturated_pixels = 0
            if dq is not None:
                group_dq = dq[:, group_idx, :, :]
                # Count pixels with DO_NOT_USE flag set
                saturated_pixels = np.sum((group_dq & DO_NOT_USE) != 0)
            else:
                # Fallback: count zero pixels if no DQ array
                saturated_pixels = np.sum(group_data == 0)
                
            saturated_counts.append(saturated_pixels)
        
        # Find where saturation effectively stops
        # We'll look for the group where the number of saturated/bad pixels
        # stops increasing significantly
        if len(saturated_counts) < 3:
            return ngroups_full
            
        # Look for the point where saturation levels off (no more significant increase)
        total_pixels = data.shape[0] * data.shape[2] * data.shape[3]
        
        # Debug: print saturation information
        print(f"DEBUG: DO_NOT_USE pixel counts by group: {saturated_counts}")
        
        # Work with actual counts instead of fractions for better sensitivity
        max_count = max(saturated_counts)
        min_count = min(saturated_counts)
        
        print(f"DEBUG: DO_NOT_USE counts - max: {max_count}, min: {min_count}")
        
        # If very little flagging overall, don't optimize
        if max_count < 10:  # Less than 10 flagged pixels ever
            print(f"DEBUG: Very low DO_NOT_USE flagging (max {max_count}), using all groups")
            return ngroups_full
        
        # Find where DO_NOT_USE counts drop significantly and stay low
        # This indicates saturation has ended
        
        # Calculate the threshold for "low" DO_NOT_USE counts
        # Use a threshold that's a small fraction of the maximum
        low_threshold = max(5, max_count * 0.1)  # 10% of max, but at least 5
        print(f"DEBUG: Low DO_NOT_USE threshold: {low_threshold}")
        
        # Find the first group where counts drop below threshold and stay low
        saturation_end_group = ngroups_full  # Default to full groups
        
        consecutive_low = 0
        required_consecutive = min(5, ngroups_full // 10)  # At least 5 consecutive low groups
        
        for i, count in enumerate(saturated_counts):
            if count <= low_threshold:
                consecutive_low += 1
                if consecutive_low >= required_consecutive:
                    # Found where saturation ends - back up to the start of the low period
                    saturation_end_group = i + 1 - consecutive_low + 1  # 1-indexed
                    print(f"DEBUG: Found {consecutive_low} consecutive low counts ending at group {i+1}")
                    break
            else:
                consecutive_low = 0
        
        # Alternative approach: look for the steepest drop in counts
        if saturation_end_group == ngroups_full and ngroups_full > 10:
            print("DEBUG: Using steepest drop approach")
            max_drop = 0
            max_drop_group = 1
            
            for i in range(1, len(saturated_counts)):
                drop = saturated_counts[i-1] - saturated_counts[i]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_group = i
            
            print(f"DEBUG: Max drop of {max_drop} at group {max_drop_group+1}")
            # Use a few groups after the steepest drop
            saturation_end_group = min(max_drop_group + 10, ngroups_full)
        
        print(f"DEBUG: Determined saturation ends at group {saturation_end_group}")
        return saturation_end_group


def remove_groups_from_file(path: str, groups_to_use: list[int] | None = None, optimize_for_desaturation: bool = True, saturation_margin: int = 2, source_type: str = 'extended') -> None:
    """
    Remove groups from FITS file, optionally optimized for desaturation.
    
    Args:
        path: Path to the FITS file
        groups_to_use: Specific list of group numbers to process (if None, processes all needed groups)
        optimize_for_desaturation: If True, only process groups needed for desaturation
        saturation_margin: Number of groups before saturation end to start processing (default: 2)
        source_type: 'extended' (process from saturation_end to total_groups) or 'point' (process from saturation_end-N to saturation_end+2)
    """
    with fits.open(path) as hdul:
        ngroups_full = hdul[0].header['NGROUPS']  #  type: ignore
        data = hdul['SCI'].data  #  type: ignore
        original_header = hdul[0].header.copy()  #  type: ignore
        # Optimize for desaturation by only processing necessary groups
        if optimize_for_desaturation:
            saturation_end_group = detect_saturation_end_group(path)
            # Start processing from saturation_margin groups before saturation ends, but at least from group 1
            min_group_to_process = max(1, saturation_end_group - saturation_margin)
            
            # Determine end group based on source type
            if source_type.lower() == 'point':
                # Point sources: only process a few groups around saturation end
                max_group_to_process = min(ngroups_full, saturation_end_group + 2)
                print(f"Point source optimization: Detected saturation ends at group {saturation_end_group}/{ngroups_full}")
                print(f"Processing groups {min_group_to_process}-{max_group_to_process-1} (focused around saturation end)")
            else:  # extended
                # Extended sources: process from saturation end to all groups
                max_group_to_process = ngroups_full
                print(f"Extended source optimization: Detected saturation ends at group {saturation_end_group}/{ngroups_full}")
                print(f"Processing groups {min_group_to_process}-{max_group_to_process-1} (skipping {min_group_to_process-1} early groups)")
        else:
            min_group_to_process = 1
            max_group_to_process = ngroups_full
        
        for ngroups in range(min_group_to_process, max_group_to_process):
            if groups_to_use is not None and ngroups not in groups_to_use:
                continue
            shape = (data.shape[0], ngroups, data.shape[2], data.shape[3])
            reduced_data = np.zeros(shape, dtype=data.dtype)
            for idx in range(ngroups):
                reduced_data[:, idx, :, :] = data[:, idx, :, :]
            reduced_data = reduced_data.astype('uint16')

            hdul['SCI'].data = reduced_data  #  type: ignore

            hdul[0].header = original_header.copy()  #  type: ignore
            hdul[0].header['NGROUPS'] = ngroups  #  type: ignore
            tools.add_header_reduction_note(
                hdul, f'Using {ngroups}/{ngroups_full} groups'
            )
            hdul[0].header['HIERARCH REMOVE_GROUP VERSION'] = (  # type: ignore
                __version__,
                'Software version',
            )

            root, filename = os.path.split(path)
            root, stage0 = os.path.split(root)
            path_out = os.path.join(
                root, 'groups', f'{ngroups}_groups', stage0, filename
            )
            tools.check_path(path_out)
            hdul.writeto(path_out, overwrite=True)
