"""
マイクで音を拾い、リアルタイムに画像を復元するプログラム
(image_to_sound.py が出した音を受信する)

使い方:
  python sound_to_image.py                  # マイクで待ち受け(ウィンドウに1行ずつ描画)
  python sound_to_image.py --list-devices   # マイクの一覧
  python sound_to_image.py --device 1       # マイクを指定
  python sound_to_image.py --file 音.wav    # マイクの代わりにWAVを読む(動作確認用)
  --no-window を付けると画面表示なしで保存だけ行います
受信が終わると received_日時.png として保存し、次の画像の待ち受けに戻ります。
"""
import argparse
import datetime
import os
import queue
import sys
import wave

import numpy as np
from scipy.signal import firwin, lfilter

from sound_protocol import (
    FS, F_SYNC, F_BLACK, F_LEADER, F_BIT1, F_BIT0,
    BIT_SEC, SYNC_SEC, HEADER_BITS,
    freq_to_value, unpack_header, channel_offset, line_duration,
)


# ---------------------------------------------------------------------------
# 音 → 「その瞬間の周波数」への変換(FM復調)
# ---------------------------------------------------------------------------
class Demodulator:
    """音声を 1700Hz 付近を中心に複素ベースバンドへ落とし、
    位相の回転速度から各サンプルの周波数を求める。ブロックをまたいで状態を保持する。"""

    def __init__(self, fs, fc=1700.0, cutoff=1000.0, ntaps=63):
        self.fs, self.fc = fs, fc
        self.taps = firwin(ntaps, cutoff, fs=fs)
        self.zi = np.zeros(ntaps - 1, dtype=complex)
        self.phase = 0.0
        self.prev = 1 + 0j

    def process(self, x):
        if len(x) == 0:
            return np.zeros(0)
        w = 2 * np.pi * self.fc / self.fs
        lo = np.exp(-1j * (self.phase + w * np.arange(len(x))))
        self.phase = (self.phase + w * len(x)) % (2 * np.pi)
        z, self.zi = lfilter(self.taps, 1.0, x * lo, zi=self.zi)
        zz = np.concatenate(([self.prev], z))
        self.prev = z[-1]
        dphi = np.angle(zz[1:] * np.conj(zz[:-1]))
        return dphi * self.fs / (2 * np.pi) + self.fc


class GrowBuffer:
    """周波数列をため込むバッファ。絶対位置(受信開始からのサンプル番号)で読める。"""

    def __init__(self):
        self.data = np.zeros(1 << 16)
        self.n = 0
        self.base = 0

    @property
    def end(self):
        return self.base + self.n

    def append(self, x):
        need = self.n + len(x)
        if need > len(self.data):
            new = np.zeros(max(need, 2 * len(self.data)))
            new[:self.n] = self.data[:self.n]
            self.data = new
        self.data[self.n:need] = x
        self.n = need

    def get(self, a, b):
        return self.data[int(a) - self.base:int(b) - self.base]

    def trim(self, keep_from):
        k = min(int(keep_from) - self.base, self.n)
        if k > 0:
            self.data[:self.n - k] = self.data[k:self.n]
            self.n -= k
            self.base += k


def moving_average(x, n):
    """長さ n の移動平均(有効部分のみ)。戻り値[i] は x[i:i+n] の平均"""
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return (cs[n:] - cs[:-n]) / n


# ---------------------------------------------------------------------------
# 受信の状態機械: 待ち受け → ヘッダ読み取り → 画像受信
# ---------------------------------------------------------------------------
class Receiver:
    def __init__(self, fs, listener, outdir="."):
        self.fs = fs
        self.ui = listener
        self.outdir = outdir
        self.buf = GrowBuffer()
        self.state = "wait"
        self.leader_pos = None

    def feed(self, freq):
        self.buf.append(freq)
        while getattr(self, "_" + self.state)():
            pass

    # --- 1. リーダー音(1900Hz)を探す ---
    def _wait(self):
        fs, buf = self.fs, self.buf
        if self.leader_pos is None:
            win = int(0.3 * fs)
            seg = buf.get(buf.base, buf.end)
            if len(seg) < win:
                return False
            inband = (np.abs(seg - F_LEADER) < 120).astype(float)
            idx = np.flatnonzero(moving_average(inband, win) >= 0.8)
            if len(idx) == 0:
                buf.trim(buf.end - win)
                return False
            # 窓の終わりは確実にリーダーの内側なので、そこから終端を探す
            self.leader_pos = buf.base + idx[0] + win
            buf.trim(self.leader_pos)
            self.ui.status("開始の合図を検出")
            return True

        # リーダーの終わり(=スタートビットの始まり)を探す
        n = int(0.010 * fs)            # 雑音で一瞬下がっただけでは反応しないよう10msで平均
        seg = buf.get(self.leader_pos, buf.end)
        if len(seg) < n:
            return False
        low = np.flatnonzero(moving_average(seg, n) < (F_LEADER + F_SYNC) / 2)
        if len(low) == 0:
            if len(seg) > 2 * fs:                      # 長すぎる → 誤検出
                self.leader_pos = None
            return False
        self.t0 = self.leader_pos + low[0] + n // 2
        self.leader_pos = None
        self.state = "header"
        return True

    # --- 2. ヘッダ(幅・高さ・画素時間)を読む ---
    def _header(self):
        fs, buf = self.fs, self.buf
        B = BIT_SEC * fs
        if buf.end < self.t0 + (HEADER_BITS + 3) * B:
            return False
        bits = []
        for k in range(HEADER_BITS):
            a = self.t0 + (1 + k + 0.2) * B
            b = self.t0 + (2 + k - 0.2) * B
            bits.append(1 if buf.get(a, b).mean() < (F_BIT1 + F_BIT0) / 2 else 0)
        hdr = unpack_header(bits)
        if hdr is None:
            self.ui.status("ヘッダの読み取りに失敗。待ち受けに戻ります")
            buf.trim(self.t0 + B)
            self.state = "wait"
            return True

        self.W, self.H, self.pt = hdr
        self.img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.line = 0
        self.miss = 0
        self.next_line = self.t0 + (HEADER_BITS + 2) * B
        self.search = 0.025 * fs          # 最初の行は広めに同期音を探す
        self.ui.start(self.img, self.pt)
        self.state = "image"
        return True

    # --- 3. 画像を1行ずつ読む ---
    def _image(self):
        fs, buf = self.fs, self.buf
        W, pt = self.W, self.pt
        L = line_duration(W, pt) * fs
        sync_n = int(round(SYNC_SEC * fs))
        exp = self.next_line
        if buf.end < exp + self.search + L + 2:
            return False

        # 予想位置の前後で同期音(1200Hz)を探し、ずれを補正する
        a = int(exp - self.search)
        seg = buf.get(a, int(exp + self.search) + sync_n)
        low = (seg < (F_SYNC + F_BLACK) / 2).astype(float)
        score = np.convolve(low, np.ones(sync_n), mode="valid")
        k = int(np.argmax(score))
        if score[k] >= 0.6 * sync_n:
            start, self.miss = a + k, 0
        else:
            start, self.miss = exp, self.miss + 1

        # R, G, B それぞれ画素ごとに周波数の平均をとる(区間の中央70%を使う)
        pn = pt * fs
        for c in range(3):
            edges = start + channel_offset(c, W, pt) * fs + np.arange(W + 1) * pn
            pa = np.floor(edges[:-1] + 0.15 * pn).astype(np.int64)
            pb = np.maximum(np.ceil(edges[1:] - 0.15 * pn).astype(np.int64), pa + 1)
            lo = pa[0]
            cs = np.concatenate(([0.0], np.cumsum(buf.get(lo, pb[-1]))))
            mean = (cs[pb - lo] - cs[pa - lo]) / (pb - pa)
            self.img[self.line, :, c] = freq_to_value(mean)

        self.ui.line(self.line, self.H)
        self.line += 1
        self.next_line = start + L
        self.search = 0.006 * fs
        buf.trim(start)

        if self.line >= self.H:
            self._finish(partial=False)
        elif self.miss >= 15:
            self.ui.status("信号が途切れました")
            self._finish(partial=True)
        return True

    def _finish(self, partial):
        from PIL import Image
        os.makedirs(self.outdir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.outdir, f"received_{stamp}{'_partial' if partial else ''}.png")
        Image.fromarray(self.img[:self.line] if partial else self.img).save(path)
        self.ui.done(path)
        self.buf.trim(self.next_line)
        self.state = "wait"


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------
class ConsoleView:
    def status(self, msg):
        print(f"\n[{msg}]", flush=True)

    def start(self, img, pt):
        h, w, _ = img.shape
        print(f"\n[受信開始] {w}x{h}, 1画素 {pt * 1000:.2f} ms", flush=True)

    def line(self, y, h):
        print(f"\r受信中 {y + 1}/{h} 行", end="", flush=True)

    def done(self, path):
        print(f"\n[保存しました] {path}\n[次の画像を待っています]", flush=True)

    def alive(self):
        return True

    def refresh(self):
        pass


class WindowView(ConsoleView):
    """matplotlib のウィンドウに受信中の画像を描く"""

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(num="Sound to Image")
        self.ax.axis("off")
        self.ax.set_title("Waiting for signal...")
        self.im = None
        self.dirty = True
        plt.show(block=False)

    def status(self, msg):
        super().status(msg)          # 日本語フォントが無い環境でも崩れないよう、画面の文字は英語
        self.ax.set_title("Signal detected..." if "検出" in msg else "Lost signal / header error")
        self.dirty = True

    def start(self, img, pt):
        super().start(img, pt)
        self.ax.clear()
        self.ax.axis("off")
        self.img = img
        self.im = self.ax.imshow(img)
        self.dirty = True

    def line(self, y, h):
        super().line(y, h)
        self.ax.set_title(f"Receiving line {y + 1}/{h}")
        self.dirty = True

    def done(self, path):
        super().done(path)
        self.ax.set_title(f"Saved {os.path.basename(path)} - waiting for next")
        self.dirty = True

    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def refresh(self):
        if self.dirty:
            if self.im is not None:
                self.im.set_data(self.img)
            self.fig.canvas.draw_idle()
            self.dirty = False
        self.plt.pause(0.03)


# ---------------------------------------------------------------------------
# 入力: マイク or WAVファイル
# ---------------------------------------------------------------------------
def run_microphone(args, view):
    import sounddevice as sd
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    demod = Demodulator(args.rate)
    rx = Receiver(args.rate, view, args.outdir)
    level = {"peak": 0.0, "blocks": 0}

    def feed(x):
        level["peak"] = max(level["peak"], float(np.abs(x).max()))
        level["blocks"] += 1
        if level["blocks"] % 110 == 0:                  # 約5秒ごと
            if level["peak"] < 3e-4:
                print("\n[警告] マイクからほぼ無音しか届いていません。"
                      "--monitor で確認してください", flush=True)
            level["peak"] = 0.0
        rx.feed(demod.process(x))

    dev = sd.query_devices(args.device if args.device is not None else sd.default.device[0])
    print(f"使用中のマイク: {dev['name']}")
    print("マイクで待ち受けています (終了: ウィンドウを閉じるか Ctrl+C)")
    with sd.InputStream(samplerate=args.rate, channels=1, dtype="float32",
                        blocksize=2048, device=args.device, callback=callback):
        while view.alive():
            try:
                feed(q.get(timeout=0.05))
                while not q.empty():
                    feed(q.get_nowait())
            except queue.Empty:
                pass
            view.refresh()


def describe_block(x, fs):
    """音量(dB)と、いちばん強い周波数を返す(診断用)"""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    db = 20 * np.log10(rms)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs > 300) & (freqs < 4000)
    peak = freqs[band][np.argmax(spec[band])]
    return db, peak


def label_for(db, peak):
    if db < -70:
        return "← ほぼ無音(マイクが無効・ミュート・別のデバイスの可能性)"
    for f, name in ((F_LEADER, "開始の合図"), (F_SYNC, "同期音/ビット"),
                    (F_BIT1, "ヘッダのビット"), (F_BIT0, "ヘッダのビット")):
        if abs(peak - f) < 60:
            return f"← {name} ({int(f)}Hz) が聞こえています"
    if F_BLACK - 30 < peak < 2300 + 30:
        return "← 画像部分の音域"
    return ""


def run_monitor(args):
    """マイクの音量と主な周波数を0.2秒ごとに表示する"""
    import sounddevice as sd
    fs = args.rate
    n = int(0.2 * fs)
    print("診断モード: マイクの音量と周波数を表示します (Ctrl+C で終了)")
    print("無音時は -60dB 以下、送信音を流すと -40〜-10dB 程度になるのが目安です\n")
    with sd.InputStream(samplerate=fs, channels=1, dtype="float32", device=args.device) as st:
        while True:
            x, _ = st.read(n)
            db, peak = describe_block(x[:, 0].astype(float), fs)
            bar = "#" * max(0, int((db + 80) / 2))
            print(f"音量 {db:6.1f} dB {bar:<40} 主な周波数 {peak:6.0f} Hz {label_for(db, peak)}",
                  flush=True)


def read_wav(path):
    with wave.open(path, "rb") as wf:
        fs, sw, ch = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(float) - 128) / 128
    elif sw == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4") / 2147483648.0
    else:
        sys.exit("対応していないWAV形式です")
    return x.reshape(-1, ch).mean(axis=1), fs


def run_file(args, view):
    x, fs = read_wav(args.file)
    demod = Demodulator(fs)
    rx = Receiver(fs, view, args.outdir)
    chunk = 4096
    for i in range(0, len(x), chunk):
        rx.feed(demod.process(x[i:i + chunk]))
        view.refresh()
    print("\nファイルの終わりに達しました")
    if isinstance(view, WindowView):
        view.plt.ioff()
        view.plt.show()


def main():
    p = argparse.ArgumentParser(description="マイクの音からリアルタイムに画像を復元します")
    p.add_argument("--file", help="マイクの代わりにこのWAVを読む")
    p.add_argument("--device", type=int, help="入力デバイス番号")
    p.add_argument("--list-devices", action="store_true", help="入力デバイスの一覧を表示")
    p.add_argument("--rate", type=int, default=FS, help="録音のサンプリング周波数")
    p.add_argument("--outdir", default=".", help="画像の保存先フォルダ")
    p.add_argument("--no-window", action="store_true", help="画面表示をしない")
    p.add_argument("--monitor", action="store_true",
                   help="診断モード: マイクの音量と周波数を表示するだけ")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.monitor:
        try:
            run_monitor(args)
        except KeyboardInterrupt:
            print("\n終了します")
        return

    view = ConsoleView() if args.no_window else WindowView()
    try:
        if args.file:
            run_file(args, view)
        else:
            run_microphone(args, view)
    except KeyboardInterrupt:
        print("\n終了します")


if __name__ == "__main__":
    main()
