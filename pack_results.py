#!/usr/bin/env python
"""
大檔傳輸打包工具：多執行緒 zstd 壓縮 + 固定大小切塊 + SHA256 校驗清單。

用途：results/ 這類數十 GB 的檔案或資料夾要從 GPU 伺服器抓回本機時，
先壓縮（SQLite 數值表/CSV 通常可壓到 1/3 以下）並切塊，之後用 sftp/scp
逐塊下載——單塊損壞或中斷只需重抓該塊，不必整檔重來。

伺服器端：
    單一檔案：Project/bin/python pack_results.py results/result.db
              → results/result.db.zst.part000, ... 與同目錄 SHA256SUMS.txt
    整個資料夾：Project/bin/python pack_results.py results
              → ./results.tar.zst.part000, ... 與 ./SHA256SUMS.txt

本機端（PowerShell，下載全部 part 檔與 SHA256SUMS.txt 後）：
    1. 校驗   ：Get-FileHash *.part* | 對照 SHA256SUMS.txt
    2. 合併   ：cmd /c copy /b results.tar.zst.part* results.tar.zst
    3. 解壓   ：pip install zstandard 後
       檔案   ：python -c "import zstandard as z,shutil; shutil.copyfileobj(z.ZstdDecompressor().stream_reader(open('result.db.zst','rb')), open('result.db','wb'), 8<<20)"
       資料夾 ：python -c "import zstandard as z,tarfile; tarfile.open(fileobj=z.ZstdDecompressor().stream_reader(open('results.tar.zst','rb')), mode='r|').extractall('.')"
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

import zstandard


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"G": 1 << 30, "M": 1 << 20, "K": 1 << 10}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


class ChunkWriter:
    """把壓縮輸出串流依 chunk_size 切成 .partNNN 檔，並累計各塊 SHA256。"""

    def __init__(self, prefix: str, chunk_size: int):
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.part_idx = -1
        self.part_bytes = 0
        self.total_bytes = 0
        self.fh = None
        self.hasher = None
        self.sums = []  # [(filename, sha256hex)]

    def _open_next(self):
        self._close_part()
        self.part_idx += 1
        path = f"{self.prefix}.part{self.part_idx:03d}"
        self.fh = open(path, "wb")
        self.hasher = hashlib.sha256()
        self.part_bytes = 0

    def _close_part(self):
        if self.fh is not None:
            self.fh.close()
            name = os.path.basename(f"{self.prefix}.part{self.part_idx:03d}")
            self.sums.append((name, self.hasher.hexdigest()))
            self.fh = None

    def write(self, data: bytes):
        while data:
            if self.fh is None or self.part_bytes >= self.chunk_size:
                self._open_next()
            room = self.chunk_size - self.part_bytes
            piece = data[:room]
            self.fh.write(piece)
            self.hasher.update(piece)
            self.part_bytes += len(piece)
            self.total_bytes += len(piece)
            data = data[room:]

    def close(self):
        self._close_part()
        sums_path = os.path.join(os.path.dirname(self.prefix) or ".", "SHA256SUMS.txt")
        with open(sums_path, "w") as f:
            for name, digest in self.sums:
                f.write(f"{digest}  {name}\n")
        return sums_path


def main():
    ap = argparse.ArgumentParser(description="zstd 壓縮 + 切塊 + SHA256（供大檔斷點下載）")
    ap.add_argument("src", help="來源檔案，例如 results/result.db")
    ap.add_argument("--chunk-size", default="2G", help="切塊大小（預設 2G）")
    ap.add_argument("--level", type=int, default=3, help="zstd 壓縮等級（預設 3）")
    ap.add_argument("--threads", type=int, default=-1, help="壓縮執行緒數（預設 -1 = 全部核心）")
    args = ap.parse_args()

    src = args.src.rstrip("/")
    is_dir = os.path.isdir(src)
    chunk_size = parse_size(args.chunk_size)

    if is_dir:
        # 資料夾走 tar 串流；總量以 du 估計（tar 標頭誤差可忽略），輸出寫在來源外層
        total = int(subprocess.check_output(["du", "-sb", src]).split()[0])
        prefix = os.path.basename(src) + ".tar.zst"
        tar_proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", os.path.dirname(src) or ".", os.path.basename(src)],
            stdout=subprocess.PIPE,
        )
        fin = tar_proc.stdout
    else:
        total = os.path.getsize(src)
        prefix = src + ".zst"
        tar_proc = None
        fin = open(src, "rb")

    cctx = zstandard.ZstdCompressor(level=args.level, threads=args.threads)
    # 資料夾模式 total 只是 du 估計值，不可寫入 zstd frame header
    # （宣告大小與實際串流不符會讓解壓端誤判資料損毀），僅供進度顯示
    cobj = cctx.compressobj() if is_dir else cctx.compressobj(size=total)
    writer = ChunkWriter(prefix, chunk_size)

    done = 0
    start = last_print = time.time()
    while True:
        block = fin.read(32 << 20)
        if not block:
            break
        writer.write(cobj.compress(block))
        done += len(block)
        now = time.time()
        if now - last_print >= 1.0:
            speed = done / (now - start) / (1 << 20)
            eta = (total - done) / max(done / (now - start), 1)
            print(f"\r  壓縮中 {min(done / total * 100, 100):5.1f}% | "
                  f"{done / (1 << 30):.2f}/{total / (1 << 30):.2f} GB | "
                  f"{speed:,.0f} MB/s | ETA {eta:,.0f}s", end="", flush=True)
            last_print = now
    fin.close()
    if tar_proc is not None and tar_proc.wait() != 0:
        sys.exit(f"tar 失敗（exit {tar_proc.returncode}），輸出的 part 檔不完整，請勿使用")
    writer.write(cobj.flush())
    sums_path = writer.close()

    elapsed = time.time() - start
    ratio = writer.total_bytes / total if total else 0
    print(f"\n完成：{total / (1 << 30):.2f} GB → {writer.total_bytes / (1 << 30):.2f} GB "
          f"（{ratio * 100:.1f}%），{elapsed:.0f}s，共 {writer.part_idx + 1} 塊")
    print(f"校驗清單：{os.path.abspath(sums_path)}")
    print("\n本機下載（PowerShell）：")
    print(f"  sftp yzu-gpu-server 後逐塊 get（中斷改用 reget），或：")
    print(f"  scp yzu-gpu-server:{os.path.abspath(prefix)}.part* .")
    print(f"  scp yzu-gpu-server:{os.path.abspath(sums_path)} .")
    print("合併＋校驗＋解壓指令見本檔 docstring。")


if __name__ == "__main__":
    main()
