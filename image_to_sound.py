"""
画像を音に変換するプログラム(マイクで受信できる方式)

使い方:
  python image_to_sound.py 画像.png -o 出力.wav     # WAVに保存
     python image_to_sound.py 画像.jpg --play         # その場でスピーカーから再生
  オプション: --size 320x240 (最大サイズ)  --pixel-ms 0.5 (1画素の長さ)
"""
import argparse
import wave

import numpy as np
from PIL import Image

from sound_protocol import (
    FS, F_SYNC, F_BLACK, F_LEADER, F_BIT1, F_BIT0,
    LEADER_SEC, BIT_SEC, SYNC_SEC, PORCH_SEC, SEP_SEC, PIXEL_TIME_UNIT,
    value_to_freq, pack_header, line_duration,
)


def load_image(path, max_w, max_h):
    img = Image.open(path).convert("RGB")      # 透明度は送れないのでRGBにする
    img.thumbnail((max_w, max_h), Image.LANCZOS)  # 縦横比を保って縮小
    return np.asarray(img)


def build_signal(rgb, pixel_units, fs=FS):
    h, w, _ = rgb.shape
    pixel_sec = pixel_units * PIXEL_TIME_UNIT
    freqs, durs = [], []

    def add(f, d):
        freqs.append(np.atleast_1d(np.asarray(f, dtype=float)))
        durs.append(np.broadcast_to(np.asarray(d, dtype=float), freqs[-1].shape))

    # --- 開始の合図とヘッダ ---
    add(F_LEADER, LEADER_SEC)
    add(F_SYNC, BIT_SEC)                                   # スタートビット
    bits = np.array(pack_header(w, h, pixel_units))
    add(np.where(bits == 1, F_BIT1, F_BIT0), BIT_SEC)
    add(F_BLACK, BIT_SEC)                                  # ストップビット

    # --- 画像本体(1行ずつ) ---
    for y in range(h):
        add([F_SYNC, F_BLACK], [SYNC_SEC, PORCH_SEC])
        for c in range(3):
            add(value_to_freq(rgb[y, :, c]), pixel_sec)
            add(F_BLACK, SEP_SEC)

    freqs = np.concatenate(freqs)
    durs = np.concatenate(durs)

    # 各区間のサンプル数(誤差が溜まらないよう累積時間から丸める)
    ends = np.round(np.cumsum(durs) * fs).astype(np.int64)
    counts = np.diff(np.concatenate(([0], ends)))
    inst_freq = np.repeat(freqs, counts)

    # 周波数を積分して位相が連続な正弦波を作る(ブツブツ音を防ぐ)
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    signal = 0.5 * np.sin(phase)

    silence = np.zeros(int(0.3 * fs))
    return np.concatenate((silence, signal, silence)).astype(np.float32)


def save_wav(path, signal, fs=FS):
    data = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data.tobytes())


def main():
    p = argparse.ArgumentParser(description="画像を音に変換します")
    p.add_argument("image", help="入力画像")
    p.add_argument("-o", "--output", help="出力WAVファイル")
    p.add_argument("--play", action="store_true", help="スピーカーから再生する")
    p.add_argument("--size", default="320x240", help="最大サイズ 幅x高さ (既定 320x240)")
    p.add_argument("--pixel-ms", type=float, default=0.5,
                   help="1画素あたりの時間[ms] 短いほど速いが崩れやすい (既定 0.5)")
    args = p.parse_args()
    if not args.output and not args.play:
        p.error("-o か --play のどちらかを指定してください")

    max_w, max_h = (int(v) for v in args.size.lower().split("x"))
    units = int(round(args.pixel_ms * 1e-3 / PIXEL_TIME_UNIT))
    if not 1 <= units <= 255:
        p.error("--pixel-ms は 0.01〜2.55 の範囲で指定してください")

    rgb = load_image(args.image, max_w, max_h)
    h, w, _ = rgb.shape
    signal = build_signal(rgb, units)
    print(f"画像 {w}x{h} → 音声 {len(signal) / FS:.1f} 秒 "
          f"(1行 {line_duration(w, units * PIXEL_TIME_UNIT) * 1000:.0f} ms)")

    if args.output:
        save_wav(args.output, signal)
        print(f"保存しました: {args.output}")
    if args.play:
        import sounddevice as sd
        print("再生中... (Ctrl+C で中止)")
        sd.play(signal, FS)
        sd.wait()


if __name__ == "__main__":
    main()
