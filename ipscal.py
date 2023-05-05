import socketio
import eventlet
import numpy as np  # NUMPY

sio = socketio.Server(async_mode='eventlet')
app = socketio.WSGIApp(sio)


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
            if check_diff[num] > 0:  # If differance value is over average differance value, replace to linear data
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


@sio.on('cal_req')
def handle_cal_req(sid, data: dict):

    raw_rssi = data.get("rssi_raw_data")
    raw_pos = data.get("pos_data")
    raw_preset = data.get("preset_data")

    print(f"Received Data from {sid}")
    print(f"Rssi Raw Data : {raw_rssi}")
    print(f"Position Data : {raw_pos}")
    print(f"Preset Data : {raw_preset}")

    if raw_rssi is not None and raw_pos is not None and raw_preset is not None:
        rssi = np.array(raw_rssi)
        pos = np.array(raw_pos)
        preset = np.array(raw_preset)
    else:
        print(f"Received Data ERROR")
        return None

    filtered_rssi_data = linear_calibration(rssi)
    calculated_pos = position_calculate(rssi, pos, preset, MAX_TRUST_DISTANCE)

    print(f"Result = {calculated_pos}")
    sio.emit("cal_res", calculated_pos.tolist())


# PRESET
MAX_TRUST_DISTANCE = 5.500  # meter
MAX_ERROR = 2.223  # meter

if __name__ == "__main__":
    print("IPSCAL RUN!!")
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app, log_output=True)
