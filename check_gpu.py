import torch
import time

print("=== PyTorch Info ===")
print("version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"\n--- Device {i} ---")
    print("name:", torch.cuda.get_device_name(i))
    p = torch.cuda.get_device_properties(i)
    print("total_memory_GB:", p.total_memory / 1024**3)
    print("multi_processor_count:", p.multi_processor_count)
    for attr in sorted(dir(p)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(p, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

print("\n=== Matmul Benchmark ===")
for n in [1024, 2048, 4096, 8192]:
    try:
        x = torch.randn(n, n, device="cuda", dtype=torch.float16)
        w = torch.randn(n, n, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        y = x @ w
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        tflops = 2.0 * n**3 / (ms / 1000) / 1e12
        print(f"  {n}x{n} fp16 matmul: {ms:.1f}ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"  {n}x{n} OOM/error: {e}")

print("\n=== Gather vs Dense Permute Benchmark ===")
for N in [27, 64, 125, 216]:
    for B, D in [(64, 128), (256, 512)]:
        try:
            x2 = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
            perm = torch.randperm(N, device="cuda")
            indices = perm.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)

            # gather
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = torch.gather(x2, 1, indices)
            torch.cuda.synchronize()
            gather_ms = (time.time() - t0) * 1000 / 50

            # dense permute (x @ P^T)
            P = torch.zeros(N, N, device="cuda", dtype=torch.float16)
            P[torch.arange(N), perm] = 1.0
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = torch.matmul(x2.transpose(1,2), P.T).transpose(1,2)
            torch.cuda.synchronize()
            dense_ms = (time.time() - t0) * 1000 / 50

            print(f"  N={N:>4} B={B:>4} D={D:>4} | gather={gather_ms:.4f}ms dense_perm={dense_ms:.4f}ms | dense/gather={dense_ms/gather_ms:.2f}x")
        except Exception as e:
            print(f"  N={N:>4} B={B:>4} D={D:>4} error: {e}")
            break
    else:
        continue
    break

print("\n=== Sparse Permute Benchmark ===")
for N in [27, 64, 125]:
    B, D = 64, 128
    try:
        x2 = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
        perm = torch.randperm(N, device="cuda")
        indices = torch.stack([torch.arange(N, device="cuda"), perm])
        vals = torch.ones(N, device="cuda", dtype=torch.float16)
        P_sp = torch.sparse_coo_tensor(indices, vals, (N, N)).coalesce()

        x_flat = x2.transpose(0,1).reshape(N, B*D)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            _ = torch.sparse.mm(P_sp, x_flat)
        torch.cuda.synchronize()
        sparse_ms = (time.time() - t0) * 1000 / 50
        print(f"  N={N:>4} sparse_perm: {sparse_ms:.4f}ms")
    except Exception as e:
        print(f"  N={N:>4} sparse error: {e}")
        break

print("\nDone.")
