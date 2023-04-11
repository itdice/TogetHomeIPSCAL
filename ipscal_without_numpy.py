import sys

def rssi_converter(rssi_data):
    m = len(rssi_data)
    n = len(rssi_data[0])

    if m > 1:
        print("ERROR // Data after the first row is discarded because the input data is too large.")

    converted_data = [[i, rssi_data[0][i]] for i in range(n)]

    return converted_data

def dot_product(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def qr_decomposition(A):
    A = [row[:] for row in A]
    m, n = len(A), len(A[0])

    Q = [[0] * m for _ in range(m)]
    R = [[0] * n for _ in range(m)]

    for k in range(n):
        R[k][k] = ((sum([A[i][k] ** 2 for i in range(k, m)])) ** 0.5)
        for i in range(k, m):
            Q[i][k] = A[i][k] / R[k][k]
        for j in range(k + 1, n):
            R[k][j] = dot_product((Q[i][k] for i in range(k, m)), (A[i][j] for i in range(k, m)))
            for i in range(k, m):
                A[i][j] -= R[k][j] * Q[i][k]

    return Q, R

def back_substitution(U, b):
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - dot_product(U[i][i + 1:], x[i + 1:])) / U[i][i]
    return x

def linear_conversion(converted_data):
    m, n = len(converted_data), len(converted_data[0])
    A = [[converted_data[i][0], 1] for i in range(m)]
    b = [converted_data[i][1] for i in range(m)]

    Q, R = qr_decomposition(A)
    b_hat = dot_product(transpose(Q), b)

    R_upper = [row[:n] for row in R[:n]]
    b_upper = b_hat[:n]

    slope, intercept = back_substitution(R_upper, b_upper)

    return slope, intercept

def linear_calibration(rssi_raw_data):
    m, n = len(rssi_raw_data), len(rssi_raw_data[0])
    rssi_cal_data = [row[:] for row in rssi_raw_data]

    for cir in range(m):
        converted_data = rssi_converter([rssi_raw_data[cir]])  # Convert Data
        slope, intercept = linear_conversion(converted_data)  # QR Decomposition and Linear Conversion

        linear_data = [i * slope + intercept for i in range(n)]  # Create Linear Data
        all_diff = [abs(rssi_raw_data[cir][i] - linear_data[i]) for i in range(n)]  # All Diffrance raw <-> linear
        average_diff = sum(all_diff) / n  # Average Diffrance

        check_diff = [all_diff[i] - average_diff for i in range(n)]

        for num in range(n):
            if check_diff[num] > 0:  # If differance value is over average differance value, replace to linear data
                rssi_cal_data[cir][num] = linear_data[num]

    return rssi_cal_data

def distance(rssi, preset):
    rssi_cal_data = [[sum(rssi[i]) / len(rssi[i]), max(rssi[i]), min(rssi[i])] for i in range(len(rssi))]

    upper_avg = [preset[i][0] - rssi_cal_data[i][0] for i in range(len(preset))]
    upper_max = [preset[i][0] - rssi_cal_data[i][1] for i in range(len(preset))]
    upper_min = [preset[i][0] - rssi_cal_data[i][2] for i in range(len(preset))]

    dis = [10 ** (upper_avg[i] / 20) for i in range(len(upper_avg))]
    error = [10 ** (upper_min[i] / 20) - 10 ** (upper_max[i] / 20) for i in range(len(upper_min))]
    valid = [1 if error[i] < MAX_TRUST_DISTANCE else 0 for i in range(len(error))]

    result_data = [[dis[i], error[i], valid[i]] for i in range(len(dis))]

    return result_data

def position_calculate(rssi, pos, preset, trust_distance):
    dis = distance(rssi, preset)  # calculate distance from beacon

    A = 2 * (pos[1][0] - pos[0][0])
    B = 2 * (pos[1][1] - pos[0][1])
    C = pos[0][0] ** 2 + pos[0][1] ** 2 - pos[1][0] ** 2 - pos[1][1] ** 2 - dis[0][0] ** 2 + dis[1][0] ** 2
    D = 2 * (pos[2][0] - pos[1][0])
    E = 2 * (pos[2][1] - pos[1][1])
    F = pos[1][0] ** 2 + pos[1][1] ** 2 - pos[2][0] ** 2 - pos[2][1] ** 2 - dis[1][0] ** 2 + dis[2][0] ** 2

    result_pos = [0, 0]
    result_pos[0] = ((B * F) - (E * C)) / ((A * E) - (D * B))
    result_pos[1] = (-1 * (A / B)) * ((B * F) - (E * C)) / ((A * E) - (D * B)) - (C / B)

    return result_pos

def main(argv):
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

    pos_data = [_posA, _posB, _posC]
    MAX_TRUST_DISTANCE = _maxTD
    MAX_ERROR = _maxER
    preset_data = [_preA, _preB, _preC]
    rssi_raw_data = [_rssA, _rssB, _rssC]

    filtered_rssi_data = linear_calibration(rssi_raw_data)
    calculated_pos = position_calculate(filtered_rssi_data, pos_data, preset_data, MAX_TRUST_DISTANCE)

    print(calculated_pos[0])
    print(calculated_pos[1])


# PRESET
MAX_TRUST_DISTANCE = 5.500  # meter
MAX_ERROR = 2.223  # meter

if __name__ == "__main__":
    main(sys.argv)
