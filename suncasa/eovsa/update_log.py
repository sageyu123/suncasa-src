from astropy.time import Time

# Timestamp for the EOVSA15 upgrade:
# On July 8, 2025, the EOVSA array was upgraded by replacing 5 old equtorial antennas with 7 new AZ/EL ones,
# marking a transition to the EOVSA15 configuration.
EOVSA15_UPGRADE_DATE = Time('2025-05-22 12:00:00')


# Timestamp for the DCM IF filter upgrade:
# On February 22, 2019, new intermediate-frequency (IF) filters were installed in the DCMs,
# providing a clean passband from 825–1150 MHz. This enabled harmonic tuning at 325 MHz intervals
# across 1–18 GHz with 50 usable bands, avoiding interference and ensuring full spectral coverage.
DCM_IF_FILTER_UPGRADE_DATE = Time('2019-02-22 12:00:00')
