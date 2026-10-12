"""
画像⇔音 変換の共通ルール(送信側・受信側の両方がこのファイルを使う)

SSTV(アマチュア無線の静止画伝送)と同じ考え方で、
  ・画素の明るさ → 音の高さ(1500Hz=黒 〜 2300Hz=白)
  ・各行の先頭 → 1200Hz の短い同期音
で表します。音の「高さ」で情報を運ぶので、スピーカー→マイクを通して
音量が変わったり雑音が乗ったりしても画像を読み取れます。

信号の並び:
  [リーダー 1900Hz 0.5秒]
  [スタートビット 1200Hz]
  [ヘッダ 48ビット: 幅, 高さ, 1画素の時間, チェックサム]  1=1100Hz / 0=1300Hz
  [ストップビット 1500Hz]
  [1行目][2行目]...  各行 = 同期1200Hz → 1500Hz → R列 → 区切り → G列 → 区切り → B列 → 区切り
"""
import numpy as np

FS = 44100                 # サンプリング周波数

F_SYNC = 1200.0            # 同期音
F_BLACK = 1500.0           # 画素値 0
F_WHITE = 2300.0           # 画素値 255
F_LEADER = 1900.0          # 送信開始の合図
F_BIT1 = 1100.0            # ヘッダのビット 1
F_BIT0 = 1300.0            # ヘッダのビット 0

LEADER_SEC = 0.5
BIT_SEC = 0.030
SYNC_SEC = 0.006
PORCH_SEC = 0.001
SEP_SEC = 0.0005
HEADER_BITS = 48
PIXEL_TIME_UNIT = 1e-5     # 1画素の時間はヘッダ内で 10µs 単位で送る


def value_to_freq(v):
    return F_BLACK + (F_WHITE - F_BLACK) * np.asarray(v, dtype=float) / 255.0


def freq_to_value(f):
    v = (np.asarray(f, dtype=float) - F_BLACK) / (F_WHITE - F_BLACK) * 255.0
    return np.clip(v, 0, 255).round().astype(np.uint8)


def pack_header(width, height, pixel_units):
    data = [width & 0xFF, width >> 8, height & 0xFF, height >> 8, pixel_units]
    data.append(sum(data) & 0xFF)                       # チェックサム
    return [(byte >> i) & 1 for byte in data for i in range(8)]


def unpack_header(bits):
    data = [sum(bits[k * 8 + i] << i for i in range(8)) for k in range(6)]
    if sum(data[:5]) & 0xFF != data[5]:
        return None
    width = data[0] | (data[1] << 8)
    height = data[2] | (data[3] << 8)
    units = data[4]
    if not (0 < width <= 2048 and 0 < height <= 2048 and units > 0):
        return None
    return width, height, units * PIXEL_TIME_UNIT


def channel_offset(c, width, pixel_sec):
    """行の先頭(同期音の始まり)から、色 c の1画素目までの時間"""
    return SYNC_SEC + PORCH_SEC + c * (width * pixel_sec + SEP_SEC)


def line_duration(width, pixel_sec):
    return SYNC_SEC + PORCH_SEC + 3 * (width * pixel_sec + SEP_SEC)
