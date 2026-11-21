"""Self-contained Momduck / Feetech STS3215 7.4 V BAM constants.

Motor identification: Rhoban/bam, bam/params/feetech_sts3215_7_4V/m1.json.
Firmware controller: Rhoban/bam, bam/feetech/actuator.py.
No import of the optional external `bam` package is required.
"""

from __future__ import annotations

STS3215_KP_FW = 32.0
STS3215_ERROR_GAIN = 0.166
STS3215_MAX_PWM = 0.97
STS3215_NOMINAL_VIN = 7.4

# Momduck training battery model (the old (7,0, 8.2) typo is fixed here).
MOMDUCK_VIN_RANGE = (7.0, 8.2)
MOMDUCK_VIN_MIN = 6.0
MOMDUCK_DELAY_MIN_LAG = 0
MOMDUCK_DELAY_MAX_LAG = 0
MOMDUCK_FRICTION_SCALE_RANGE = (0.9, 1.1)

# Identified STS3215 7.4 V m1 parameters. q_offset is a testbench calibration,
# not a property to apply to the robot's joint positions.
STS3215_M1_PARAMS = {
    "kt": 1.1776311631627974,
    "error_gain_ratio": 1.0049367060648848,
    "R": 2.478681934101461,
    "armature": 0.02608291141307205,
    "max_velocity": 5.2883481319359,
    "friction_base": 0.05152047758105413,
    "friction_viscous": 0.05989161871380604,
}

__all__ = [
    "MOMDUCK_DELAY_MAX_LAG",
    "MOMDUCK_DELAY_MIN_LAG",
    "MOMDUCK_FRICTION_SCALE_RANGE",
    "MOMDUCK_VIN_MIN",
    "MOMDUCK_VIN_RANGE",
    "STS3215_ERROR_GAIN",
    "STS3215_KP_FW",
    "STS3215_M1_PARAMS",
    "STS3215_MAX_PWM",
    "STS3215_NOMINAL_VIN",
]
