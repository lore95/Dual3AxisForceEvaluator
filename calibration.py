# Force conversion constants
# from sensor to pcb:
#   V1 -> Fx
#   V2 -> Fy
#   V3 -> Fz
# =========================

VREF = 1.2
GAIN = 128
ADC_FULL_SCALE = 2**23  # signed bipolar ADC

SENSORDIFFERENTIAL = 10/3.3
# Right sensor sensitivities (mV/V) which is the left ont he software provide 
# to be checked on calibration hbm certificaiton
SENS_RX = 1.056
SENS_RY = 0.981
SENS_RZ = 0.981

# Rated loads (N)
FX_RATED = 5000.0
FY_RATED = 5000.0
FZ_RATED = 20000.0



# =========================
# Conversion helpers
# =========================

def adc_to_sensor_voltage(adc_value, adc_zero):
    """
    Convert raw ADC counts to differential sensor voltage in Volts.
    """
    return (adc_value - adc_zero) * VREF / (GAIN * ADC_FULL_SCALE)


def sensor_voltage_to_force(v_sensor, sensitivity_mV_per_V, rated_force_N, vexc):
    """
    Convert differential sensor voltage to force in Newtons.
    """
    v_mV = v_sensor * 1000.0 * SENSORDIFFERENTIAL
    return (v_mV / (sensitivity_mV_per_V * vexc)) * rated_force_N


def right_sensor_counts_to_newtons(v1_raw, v2_raw, v3_raw, zero_offsets, vexc):
    """
    Convert raw ADC counts to forces in Newtons for the RIGHT sensor.
    Mapping:
      v1 -> Fx
      v2 -> Fy
      v3 -> Fz
    """
    z1, z2, z3 = zero_offsets

    vx = adc_to_sensor_voltage(v1_raw, z1)
    vy = adc_to_sensor_voltage(v2_raw, z2)
    vz = adc_to_sensor_voltage(v3_raw, z3)

    fx = sensor_voltage_to_force(vx, SENS_RX, FX_RATED, vexc)
    fy = sensor_voltage_to_force(vy, SENS_RY, FY_RATED, vexc)
    fz = sensor_voltage_to_force(vz, SENS_RZ, FZ_RATED, vexc)

    return fx, fy, fz