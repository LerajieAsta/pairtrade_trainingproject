"""
H200 GPU 效能壓力測試
======================================================================
用法：
    python benchmark_h200.py               # 快速全套（~2 分鐘）
    python benchmark_h200.py --sustain 10  # 追加 10 分鐘持續滿載（燒機/散熱驗證）
    python benchmark_h200.py --vram 0.90   # VRAM 佔用目標（預設 0.85）

監控（另開終端）：
    nvidia-smi dmon -s pucm      # 功耗/利用率/時脈/記憶體 逐秒
    watch -n 1 nvidia-smi        # 總覽

H200 SXM 參考峰值（dense）：BF16 ≈ 989 TFLOPS、TF32 ≈ 494、FP32 ≈ 67；
HBM3e 頻寬 4.8 TB/s；VRAM 141 GB。實測 GEMM 通常可達峰值 70–85%。
"""

import argparse
import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gemm_tflops(dtype: torch.dtype, size: int = 8192, iters: int = 30) -> float:
    """大型方陣矩陣乘吞吐（TFLOPS）——計算峰值測試。"""
    dev = "cuda"
    a = torch.randn(size, size, device=dev, dtype=dtype)
    b = torch.randn(size, size, device=dev, dtype=dtype)
    for _ in range(5):                      # warmup
        _ = a @ b
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    _sync()
    sec = time.perf_counter() - t0
    return 2 * size**3 * iters / sec / 1e12


def membw_gbs(numel: int = 2_000_000_000, iters: int = 20) -> float:
    """大張量拷貝頻寬（GB/s）——HBM 頻寬測試（讀+寫）。"""
    x = torch.randn(numel // 2, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    _sync()
    sec = time.perf_counter() - t0
    return x.numel() * 2 * 2 * iters / sec / 1e9   # fp16=2B, 讀+寫

def lstm_train_throughput(hidden: int = 1024, layers: int = 4, batch: int = 8192,
                          seq: int = 10, feat: int = 6, steps: int = 50) -> float:
    """
    LSTM 訓練吞吐（samples/s）——模擬 DRL-FQI 的實際負載形狀
    （forward + backward + Adam step）。
    """
    model = torch.nn.LSTM(feat, hidden, layers, batch_first=True).cuda()
    head = torch.nn.Linear(hidden, 3).cuda()
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    x = torch.randn(batch, seq, feat, device="cuda")
    y = torch.randn(batch, 3, device="cuda")
    loss_fn = torch.nn.MSELoss()
    for _ in range(5):
        out, _ = model(x)
        loss = loss_fn(head(out[:, -1]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        out, _ = model(x)
        loss = loss_fn(head(out[:, -1]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    _sync()
    return batch * steps / (time.perf_counter() - t0)


def vram_fill(target_frac: float = 0.85) -> float:
    """填充 VRAM 至目標比例，回傳實際佔用 GB（結束後釋放）。"""
    total = torch.cuda.get_device_properties(0).total_memory
    blocks, block_bytes = [], 1 << 30                      # 1 GiB / 塊
    try:
        while torch.cuda.memory_allocated() < total * target_frac:
            blocks.append(torch.empty(block_bytes // 2, device="cuda", dtype=torch.float16))
    except torch.cuda.OutOfMemoryError:
        pass
    used = torch.cuda.memory_allocated() / 1e9
    del blocks
    torch.cuda.empty_cache()
    return used


def sustained_load(minutes: float, vram_frac: float = 0.85):
    """持續滿載：常駐 VRAM 佔用 + GEMM/LSTM 交替，直到時間到。"""
    total = torch.cuda.get_device_properties(0).total_memory
    blocks = []
    try:
        while torch.cuda.memory_allocated() < total * vram_frac:
            blocks.append(torch.empty((1 << 30) // 2, device="cuda", dtype=torch.float16))
    except torch.cuda.OutOfMemoryError:
        pass
    print(f"  常駐 VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    model = torch.nn.LSTM(6, 512, 2, batch_first=True).cuda()
    head = torch.nn.Linear(512, 3).cuda()
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()))
    x = torch.randn(4096, 10, 6, device="cuda")
    y = torch.randn(4096, 3, device="cuda")

    deadline = time.time() + minutes * 60
    n = 0
    while time.time() < deadline:
        for _ in range(20):
            _ = a @ b
        out, _ = model(x)
        loss = torch.nn.functional.mse_loss(head(out[:, -1]), y)
        opt.zero_grad(); loss.backward(); opt.step()
        n += 1
        if n % 50 == 0:
            _sync()
            print(f"  ...{(deadline - time.time())/60:.1f} 分鐘剩餘")
    _sync()


def main():
    ap = argparse.ArgumentParser(description="H200 GPU stress benchmark")
    ap.add_argument("--sustain", type=float, default=0.0, help="持續滿載分鐘數（燒機）")
    ap.add_argument("--vram", type=float, default=0.85, help="VRAM 佔用目標比例")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("找不到 CUDA GPU。")

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM {p.total_memory/1e9:.0f} GB | SM {p.multi_processor_count} | "
          f"CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("\n[1/4] GEMM 計算峰值")
    for name, dt in [("FP32/TF32", torch.float32), ("BF16", torch.bfloat16), ("FP16", torch.float16)]:
        print(f"  {name:10s}: {gemm_tflops(dt):8.1f} TFLOPS")

    print("[2/4] HBM 頻寬")
    print(f"  copy      : {membw_gbs():8.0f} GB/s")

    print("[3/4] LSTM 訓練吞吐（DRL-FQI 負載形狀）")
    for batch in (4096, 16384, 65536):
        print(f"  batch {batch:6d}: {lstm_train_throughput(batch=batch):12,.0f} samples/s")

    print(f"[4/4] VRAM 填充（目標 {args.vram:.0%}）")
    print(f"  達成: {vram_fill(args.vram):.1f} GB")

    if args.sustain > 0:
        print(f"\n[持續滿載 {args.sustain:.0f} 分鐘] （nvidia-smi dmon 觀察功耗/溫度）")
        sustained_load(args.sustain, args.vram)
    print("\n完成。")


if __name__ == "__main__":
    main()
