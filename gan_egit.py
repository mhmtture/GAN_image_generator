"""
Tüm yüz veri setiyle DCGAN eğitimi: rastgele gürültüden her seferinde FARKLI, sentetik bir insan yüzü üretir.

Mimari
  Generator (Üretici)     : z (128 boyutlu rastgele vektör) -> 64x64x3 yüz görüntüsü
  Discriminator (Ayırıcı) : görüntü -> "gerçek mi sahte mi" skoru
  İkisi birbirine karşı eğitilir: G, D'yi kandırmaya; D, sahteyi yakalamaya çalışır.

Küçük veri seti (~9.6k görsel) için kararlılık teknikleri
  - DiffAugment  : D'ye giden gerçek ve sahte görüntülere aynı türden rastgele renk/kaydırma/kesme
                   uygulanır -> D veri setini ezberleyemez, G çeşitli yüzler üretmeye zorlanır.
  - Spectral Norm: D'nin ağırlıkları normalize edilir -> eğitim patlamaz.
  - Hinge loss   : GAN için kararlı kayıp fonksiyonu.
  - EMA          : G ağırlıklarının hareketli ortalaması; daha temiz örnekler üretir.

Kullanım
  python gan_egit.py                       # varsayılan: data/ klasöründeki tüm görseller, 300 epoch
  python gan_egit.py --epochs 500 --batch 128
  python gan_egit.py --resume              # son checkpoint'ten devam et
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

NZ = 128  # gürültü vektörü boyutu


# ----------------------------------------------------------------------------
# Veri
# ----------------------------------------------------------------------------
def load_faces(data_dir: str, size: int, cache: str) -> np.ndarray:
    """Tüm görselleri size x size'a getirip tek bir uint8 dizi olarak döndürür (önbellekli)."""
    if os.path.exists(cache):
        arr = np.load(cache)
        if arr.shape[1] == size:
            return arr
    paths = sorted(p for ext in ("jpg", "jpeg", "png")
                   for p in glob.glob(os.path.join(data_dir, "**", f"*.{ext}"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"'{data_dir}' içinde görsel bulunamadı")
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)  # ortadan kare kırp
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        imgs.append(np.asarray(im.resize((size, size), Image.LANCZOS)))
    arr = np.stack(imgs)  # N, H, W, 3
    np.save(cache, arr)
    return arr


# ----------------------------------------------------------------------------
# DiffAugment (Zhao et al., 2020) - farklılaştırılabilir veri artırma
# ----------------------------------------------------------------------------
def diff_augment(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    dev = x.device
    # renk: parlaklık, doygunluk, kontrast
    x = x + (torch.rand(n, 1, 1, 1, device=dev) - 0.5)
    m = x.mean(dim=1, keepdim=True)
    x = (x - m) * (torch.rand(n, 1, 1, 1, device=dev) * 2) + m
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - m) * (torch.rand(n, 1, 1, 1, device=dev) + 0.5) + m
    # kaydırma (1/8 oranında)
    h, w = x.shape[2:]
    sh, sw = h // 8, w // 8
    tx = torch.randint(-sh, sh + 1, (n, 1, 1), device=dev)
    ty = torch.randint(-sw, sw + 1, (n, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=dev), torch.arange(h, device=dev),
                                torch.arange(w, device=dev), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    xp = F.pad(x, [1, 1, 1, 1])
    x = xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    # cutout (görüntünün yarısı boyutunda bir kare silinir)
    ch, cw = h // 2, w // 2
    ox = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=dev), torch.arange(ch, device=dev),
                                torch.arange(cw, device=dev), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(n, h, w, device=dev, dtype=x.dtype)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


# ----------------------------------------------------------------------------
# Modeller (64x64)
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=64):
        super().__init__()

        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),  # 4x4
            *up(ngf * 8, ngf * 4),   # 8x8
            *up(ngf * 4, ngf * 2),   # 16x16
            *up(ngf * 2, ngf),       # 32x32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1), nn.Tanh(),  # 64x64
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        sn = nn.utils.spectral_norm

        def down(i, o):
            return [sn(nn.Conv2d(i, o, 4, 2, 1)), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(
            *down(3, ndf),            # 32x32
            *down(ndf, ndf * 2),      # 16x16
            *down(ndf * 2, ndf * 4),  # 8x8
            *down(ndf * 4, ndf * 8),  # 4x4
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1.0 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Yüz veri setiyle DCGAN eğitimi")
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--sample-every", type=int, default=10, help="Kaç epoch'ta bir örnek kaydedilsin")
    ap.add_argument("--out", default="gan_ciktilar")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.size == 64, "Bu mimari 64x64 için tasarlandı"

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.join(args.out, "ornekler"), exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.pt")

    # Veri: tamamı GPU belleğinde uint8 olarak tutulur (~120 MB) -> DataLoader gerekmez, çok hızlı
    t = time.time()
    faces = load_faces(args.data, args.size, os.path.join(args.data, f"faces_{args.size}.npy"))
    data = torch.from_numpy(faces).to(dev).permute(0, 3, 1, 2).contiguous()  # N,3,H,W uint8
    n_data = data.size(0)
    print(f"Cihaz: {dev} ({torch.cuda.get_device_name() if dev.type == 'cuda' else 'CPU'})")
    print(f"Veri: {n_data} yüz, {args.size}x{args.size}  ({time.time() - t:.1f} sn)")
    save_image(data[:64].float() / 255, os.path.join(args.out, "gercek_ornekler.png"), nrow=8)

    G, D = Generator().to(dev), Discriminator().to(dev)
    G.apply(init_weights)
    D.apply(init_weights)
    G_ema = Generator().to(dev)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval().requires_grad_(False)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Sabit gürültü: her örnek kaydında AYNI z'ler kullanılır -> aynı yüzlerin gelişimi izlenebilir
    fixed_z = torch.randn(64, NZ, device=dev, generator=torch.Generator(dev).manual_seed(1234))

    start_epoch = 1
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch = ck["epoch"] + 1
        print(f"Checkpoint yüklendi, {start_epoch}. epoch'tan devam")

    steps_per_epoch = n_data // args.batch
    log = open(os.path.join(args.out, "kayit.csv"), "a")
    if start_epoch == 1:
        log.write("epoch,lossD,lossG,D_gercek,D_sahte,sure_sn\n")

    def sample(tag):
        with torch.no_grad():
            img = G_ema(fixed_z) * 0.5 + 0.5
        save_image(img, os.path.join(args.out, "ornekler", f"epoch_{tag}.png"), nrow=8)

    if start_epoch == 1:
        sample("0000")  # eğitim öncesi: saf gürültü

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        perm = torch.randperm(n_data, device=dev)
        sD = sG = sR = sF = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            real = data[idx].float() / 127.5 - 1.0
            flip = torch.rand(real.size(0), device=dev) < 0.5  # yatay çevirme
            real = torch.where(flip[:, None, None, None], real.flip(3), real)

            # --- Discriminator adımı ---
            z = torch.randn(real.size(0), NZ, device=dev)
            with torch.no_grad():
                fake = G(z)
            d_real = D(diff_augment(real))
            d_fake = D(diff_augment(fake))
            lossD = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

            # --- Generator adımı ---
            z = torch.randn(real.size(0), NZ, device=dev)
            lossG = -D(diff_augment(G(z))).mean()
            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()
            ema_update(G_ema, G, args.ema)

            sD += lossD.item(); sG += lossG.item()
            sR += d_real.mean().item(); sF += d_fake.mean().item()

        k = steps_per_epoch
        el = time.time() - t0
        log.write(f"{epoch},{sD / k:.4f},{sG / k:.4f},{sR / k:.4f},{sF / k:.4f},{el:.0f}\n"); log.flush()
        if epoch % 5 == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:4d}/{args.epochs} | lossD {sD / k:.3f} | lossG {sG / k:.3f} | "
                  f"D(gerçek) {sR / k:+.2f} D(sahte) {sF / k:+.2f} | {el / 60:.1f} dk")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample(f"{epoch:04d}")
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(), "epoch": epoch}, ckpt_path)
            torch.save(G_ema.state_dict(), os.path.join(args.out, "generator.pt"))

    log.close()
    print(f"Bitti. Üretici model: {os.path.join(args.out, 'generator.pt')}")


if __name__ == "__main__":
    main()
