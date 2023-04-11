import sys
import numpy as np  # NUMPY


def rssi_converter(rssi_data):
    m, n = rssi_data.shape

    if m > 1:
        print("ERROR // Data after the first row is discarded because the input data is too large.")

    converted_data = np.array([np.arange(n), rssi_data[0, :]]).T

    return converted_data


def linear_conversion(converted_data):
    m, n = converted_data.shape
    A = np.array([converted_data[:, 0], np.ones(m)]).T
    b = converted_data[:, 1]

    Q, R = np.linalg.qr(A)  # QR Decomposition
    b_hat = Q.T.dot(b)

    R_upper = R[:n, :]
    b_upeer = b_hat[:n]

    slope, intercept = np.linalg.solve(R_upper, b_upeer)  # Linear Solve

    return slope, intercept


def linear_calibration(rssi_raw_data):
    m, n = rssi_raw_data.shape
    rssi_cal_data = rssi_raw_data.copy()

    for cir in range(m):
        converted_data = rssi_converter(np.array([rssi_raw_data[cir, :]]))  # Convert Data
        slope, intercept = linear_conversion(converted_data)  # QR Decomposition and Linear Conversion

        linear_data = np.arange(n) * slope + intercept  # Create Linear Data
        all_diff = abs(rssi_raw_data[cir, :] - linear_data)  # All Diffrance raw <-> linear
        average_diff = all_diff.sum() / n  # Average Diffrance

        check_diff = all_diff - average_diff

        for num in range(n):
            if (check_diff[num] > 0):  # If differance value is over average differance value, replace to linear data
                rssi_cal_data[cir, num] = linear_data[num]

    return rssi_cal_data


def distance(rssi, preset):
    rssi_cal_data = np.array([rssi.mean(axis=1), rssi.max(axis=1), rssi.min(axis=1)]).T

    upper_avg = preset[:, 0] - rssi_cal_data[:, 0]
    upper_max = preset[:, 0] - rssi_cal_data[:, 1]
    upper_min = preset[:, 0] - rssi_cal_data[:, 2]

    dis = 10 ** (upper_avg / (10 * 2))
    error = 10 ** (upper_min / (10 * 2)) - 10 ** (upper_max / (10 * 2))
    valid = np.ones(error.shape[0])

    for cir in range(error.shape[0]):  # If error value is bigger than MAX_TRUST_DISTANCE, distance value can't trust
        valid[cir] = 1 if error[cir] < MAX_TRUST_DISTANCE else 0

    result_data = np.array([dis, error, valid]).T

    return result_data


def position_calculate(rssi, pos, preset, trust_distance):
    dis = distance(rssi, preset)  # calculate distance from beacon

    A = 2 * (pos[1, 0] - pos[0, 0])
    B = 2 * (pos[1, 1] - pos[0, 1])
    C = pos[0, 0] ** 2 + pos[0, 1] ** 2 - pos[1, 0] ** 2 - pos[1, 1] ** 2 - dis[0, 0] ** 2 + dis[1, 0] ** 2
    D = 2 * (pos[2, 0] - pos[1, 0])
    E = 2 * (pos[2, 1] - pos[1, 1])
    F = pos[1, 0] ** 2 + pos[1, 1] ** 2 - pos[2, 0] ** 2 - pos[2, 1] ** 2 - dis[1, 0] ** 2 + dis[2, 0] ** 2

    result_pos = np.zeros(2)
    result_pos[0] = ((B * F) - (E * C)) / ((A * E) - (D * B))
    result_pos[1] = (-1 * (A / B)) * ((B * F) - (E * C)) / ((A * E) - (D * B)) - (C / B)

    return result_pos


'''
rssi_raw_data = np.array([[-53., -54., -54., -55., -55., -55., -56., -56., -80., -57.],
                      [-53., -54., -54., -55., -55., -55., -56., -56., -80., -57.],
                      [-53., -54., -54., -55., -55., -55., -56., -56., -80., -57.]]) # rssi raw data
pos_data = np.array([[5., 0.], [10., 8.], [0., 10.]]) # Beacon position data
preset_data = np.array([[-40.], [-42.], [-43.]]) # Beacon preset 1M rssi data

MAX_TRUST_DISTANCE = 5.500 # meter
MAX_ERROR = 2.223 # meter

filtered_rssi_data = linear_calibration(rssi_raw_data)

calculated_pos = position_calculate(filtered_rssi_data, pos_data, preset_data, MAX_TRUST_DISTANCE)
# result position

print(calculated_pos)
'''


def main(argv):
    """
    ipscal a0 a1 b0 b1 c0 c1 d0 e0 f0 f1 f2 g0 h0~ i0~ j0~
    a -> A Beacon Position Data
    b -> B Beacon Position Data
    c -> C Beacon Position Data
    d -> Max Trust Distance
    e -> Max Error
    f -> Preset rssi data
    g -> Number of rssi data
    h -> A Beacon rssi data
    i -> B Beacon rssi data
    j -> C Beacon rssi data
    All data must have decimal points
    """
    _posA = list(map(float, argv[1:3]))
    _posB = list(map(float, argv[3:5]))
    _posC = list(map(float, argv[5:7]))
    _maxTD = float(argv[7])
    _maxER = float(argv[8])
    _preA = list(map(float, argv[9:10]))
    _preB = list(map(float, argv[10:11]))
    _preC = list(map(float, argv[11:12]))
    _num = int(argv[12])
    _rssA = list(map(float, argv[13:(13 + _num)]))
    _rssB = list(map(float, argv[(13 + _num):(13 + 2 * _num)]))
    _rssC = list(map(float, argv[(13 + 2 * _num):(13 + 3 * _num)]))

    pos_data = np.array([_posA, _posB, _posC])
    MAX_TRUST_DISTANCE = _maxTD
    MAX_ERROR = _maxER
    preset_data = np.array([_preA, _preB, _preC])
    rssi_raw_data = np.array([_rssA, _rssB, _rssC])

    filtered_rssi_data = linear_calibration(rssi_raw_data)
    calculated_pos = position_calculate(filtered_rssi_data, pos_data, preset_data, MAX_TRUST_DISTANCE)

    print(calculated_pos[0])
    print(calculated_pos[1])


# PRESET
MAX_TRUST_DISTANCE = 5.500  # meter
MAX_ERROR = 2.223  # meter

if __name__ == "__main__":
    main(sys.argv)
