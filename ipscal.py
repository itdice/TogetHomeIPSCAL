import socket
import threading
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


def binder(client_socket, addr):
    print('Connected by', addr)
    try:
        # 데이터 수신은 blocking
        while True:
            # 30 byte buffer
            data_len = client_socket.recv(4)
            # 가장 앞 4 byte 는 전송할 data 의 크기 : int,  little big endian 으로 수신
            length = int.from_bytes(data_len, "little")
            # 다시 data 수신 한다.
            data = client_socket.recv(length)

            received_rssi_raw_data = np.array(
                [
                    [data[0] - 256, data[1] - 256, data[2] - 256, data[3] - 256, data[4] - 256,
                     data[5] - 256, data[6] - 256, data[7] - 256, data[8] - 256, data[9] - 256],
                    [data[10] - 256, data[11] - 256, data[12] - 256, data[13] - 256, data[14] - 256,
                     data[15] - 256, data[16] - 256, data[17] - 256, data[18] - 256, data[19] - 256],
                    [data[20] - 256, data[21] - 256, data[22] - 256, data[23] - 256, data[24] - 256,
                     data[25] - 256, data[26] - 256, data[27] - 256, data[28] - 256, data[29] - 256]
                ])  # rssi raw data

            pos_data = np.array([[data[30], data[31]],
                                 [data[32], data[33]],
                                 [data[34], data[35]]])  # Beacon position data

            preset_data = np.array(
                [[data[36] - 256], [data[37] - 256], [data[38] - 256]])  # Beacon preset 1M rssi data

            filtered_rssi_data = linear_calibration(received_rssi_raw_data)

            calculated_pos = position_calculate(filtered_rssi_data, pos_data, preset_data, MAX_TRUST_DISTANCE)
            # result position

            # debug
            print(f"Received from {addr}")
            print(calculated_pos)

            # byte 배열로 변환
            send_data = str(calculated_pos[0]) + " " + str(calculated_pos[1])
            msg = send_data.encode()

            msg_len = len(send_data)

            print(send_data)
            client_socket.sendall(msg_len.to_bytes(4, byteorder='little'))
            client_socket.sendall(msg)

    except socket.error as exc:
        print(f"[Except] (address : {addr}) (error :{exc})")
    finally:
        client_socket.close()


def main(argv):
    # socket input part
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 9999))
    server_socket.listen()

    try:
        while True:
            client_socket, addr = server_socket.accept()
            th = threading.Thread(target=binder, args=(client_socket, addr))
            # thread 를 이용 해서 client 접속 대기를 만들고 다시 다른 client 대기
            th.start()
    except socket.error as exc:
        print(f"[Except] (error :{exc})")
    finally:
        server_socket.close()


# PRESET
MAX_TRUST_DISTANCE = 5.500  # meter
MAX_ERROR = 2.223  # meter

if __name__ == "__main__":
    main(sys.argv)
