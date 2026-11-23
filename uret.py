"""
Eğitilmiş üretici (generator) ile yeni sentetik yüzler üretir.
Her çalıştırmada farklı rastgele gürültü (z) kullanıldığı için her seferinde farklı yüzler çıkar.

Kullanım
  python uret.py                    # 64 yeni yüz -> uretilen/ klasörü + toplu kolaj
  python uret.py --n 16 --seed 7    # aynı seed = aynı yüzler (tekrarlanabilir)
  python uret.py --komsu            # her üretilen yüzün veri setindeki en yakın gerçek yüzünü de göster
                                    # (modelin ezberlemediğini, yeni yüzler ürettiğini kontrol etmek için)
  python uret.py --gecis            # iki rastgele yüz arasında yumuşak geçiş (latent interpolasyon)
"""

import argparse
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image

from gan_egit import NZ, Generator


def main():
    ap = argparse.ArgumentParser(description="Eğitilmiş GAN ile yeni yüz üretimi")
    ap.add_argument("--model", default="gan_ciktilar/generator.pt")
    ap.add_argument("--n", type=int, default=64, help="Üretilecek yüz sayısı")
    ap.add_argument("--seed", type=int, default=None, help="Boş bırakılırsa her seferinde farklı")
    ap.add_argument("--out", default="uretilen")
    ap.add_argument("--buyut", type=int, default=128, help="Tekil görüntülerin kaydedileceği boyut")
    ap.add_argument("--komsu", action="store_true", help="Veri setindeki en yakın yüzle karşılaştır")
    ap.add_argument("--veri", default="data/faces_64.npy")
    ap.add_argument("--gecis", action="store_true", help="İki yüz arası geçiş görüntüsü üret")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(dev)
    G.load_state_dict(torch.load(args.model, map_location=dev))
    G.eval()

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
    gen = torch.Generator(dev).manual_seed(seed)
    run_dir = os.path.join(args.out, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.n, NZ, device=dev, generator=gen)
        imgs = (G(z) * 0.5 + 0.5).clamp(0, 1)

    # Toplu kolaj + tekil dosyalar
    nrow = int(np.ceil(np.sqrt(args.n)))
    save_image(imgs, os.path.join(run_dir, "kolaj.png"), nrow=nrow, padding=2)
    big = torch.nn.functional.interpolate(imgs, size=args.buyut, mode="bicubic", align_corners=False).clamp(0, 1)
    for i, im in enumerate(big):
        save_image(im, os.path.join(run_dir, f"yuz_{i:03d}.png"))
    print(f"Seed {seed}: {args.n} yüz üretildi -> {os.path.abspath(run_dir)}")

    # En yakın komşu kontrolü: üst satır üretilen, alt satır veri setindeki en benzer gerçek yüz
    if args.komsu:
        data = torch.from_numpy(np.load(args.veri)).to(dev).permute(0, 3, 1, 2).float() / 255
        k = min(args.n, 16)
        q = imgs[:k].reshape(k, -1)
        dist = torch.cdist(q, data.reshape(data.size(0), -1))
        nn_idx = dist.argmin(dim=1)
        save_image(torch.cat([imgs[:k], data[nn_idx]]), os.path.join(run_dir, "en_yakin_komsu.png"),
                   nrow=k, padding=2)
        print(f"En yakın komşu karşılaştırması: {os.path.join(run_dir, 'en_yakin_komsu.png')}  "
              f"(üst: üretilen, alt: veri setindeki en benzer)")

    # Latent uzayda geçiş: 2 rastgele yüz arasında 10 adım
    if args.gecis:
        rows = []
        with torch.no_grad():
            for _ in range(6):
                a, b = torch.randn(2, NZ, device=dev, generator=gen)
                t = torch.linspace(0, 1, 10, device=dev)[:, None]
                rows.append((G(a * (1 - t) + b * t) * 0.5 + 0.5).clamp(0, 1))
        save_image(torch.cat(rows), os.path.join(run_dir, "gecis.png"), nrow=10, padding=2)
        print(f"Geçiş görüntüsü: {os.path.join(run_dir, 'gecis.png')}")


if __name__ == "__main__":
    main()
