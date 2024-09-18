
# Function to convert MaxRSS to bytes
import numpy as np
import pandas as pd


def parse_maxrss(rss_str):
    if pd.isnull(rss_str):
        return np.nan
    rss_str = str(rss_str).strip()
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    for unit in units:
        if rss_str.upper().endswith(unit):
            try:
                return float(rss_str[:-1]) * units[unit]
            except ValueError:
                return np.nan
    try:
        return float(rss_str)
    except ValueError:
        return np.nan
    
    
    
def convert_maxrss_to_bytes(maxrss: str) -> int:
    """
    Converts a MaxRSS string (with 'K', 'M', or 'G' suffix) to bytes.
    
    Parameters:
    - maxrss (str): MaxRSS value as a string, e.g., '123K', '1.2G'.
    
    Returns:
    - int: MaxRSS value converted to bytes.
    """
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    
    # Strip any spaces from the string
    maxrss = maxrss.strip().upper()
    
    # Extract the numeric part and the unit part
    try:
        value = float(maxrss[:-1])
        unit = maxrss[-1]
    except ValueError:
        raise ValueError("Invalid MaxRSS format.")
    
    if unit in units:
        return int(value * units[unit])
    else:
        raise ValueError(f"Unknown unit '{unit}' in MaxRSS value.")
    
    
# Helper function to convert bytes into KB, MB, GB, etc.
def bytes_to_human_readable(num_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"