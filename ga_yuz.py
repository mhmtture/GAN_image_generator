"""
Genetik Algoritma ile hedef bir insan yüzü görüntüsüne benzeyen yapay görüntü üretimi.

Akış:
  1. Hedef görüntü yüklenir (HuggingFace "shravya11/human-faces-dataset" veya image.jpg).
  2. Rastgele piksellerden oluşan bir popülasyon oluşturulur.
  3. Her nesilde: fitness (hedefe benzerlik) -> seçilim (turnuva) -> çaprazlama -> mutasyon -> elitizm.
  4. Belirli aralıklarla en iyi birey diske kaydedilir (0. nesil = tamamen rastgele).
  5. Sonunda tüm ara görüntülerden bir ilerleme kolajı ve GIF oluşturulur.

Kullanım:
  python ga_yuz.py                          # önce dataset'i dener, olmazsa image.jpg
  python ga_yuz.py --source image           # doğrudan image.jpg
  python ga_yuz.py --source dataset --index 5 --generations 20000 --save-every 1000
"""

import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------------
# Hedef görüntüyü yükleme
# ----------------------------------------------------------------------------
def load_target_from_dataset(index: int) -> Image.Image:
    """Dataset'in tamamını indirmeden (streaming) sadece istenen görseli alır."""
    import logging
    from datasets import load_dataset

    # Streaming modunda kapanışta basılan zararsız "Task was destroyed" uyarılarını gizle
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    ds =load_dataset("shravya11/human-faces-dataset", split="train", streaming=True)
    # İterasyonu sonuna kadar tüketmek, arka plandaki stream görevlerinin düzgün kapanmasını sağlar
    samples = list(ds.skip(index).take(1))
    if not samples:
        raise IndexError(f"Dataset'te {index}. örnek yok")
    # Görsel sütununun adını otomatik bul
    for value in samples[0].values():
        if isinstance(value, Image.Image):
            return value.convert("RGB")
    raise ValueError(f"Örnekte görüntü sütunu bulunamadı: {list(samples[0].keys())}")


def load_target(source: str, index: int, image_path: str) -> tuple[Image.Image, str]:
    if source in ("auto", "dataset"):
        try:
            img = load_target_from_dataset(index)
            return img, f"dataset[{index}]"
        except Exception as e:  # ağ hatası, dataset yok vb.
            if source == "dataset":
                raise
            print(f"[!] Dataset yüklenemedi ({type(e).__name__}: {e}). '{image_path}' kullanılacak.")
    return Image.open(image_path).convert("RGB"), image_path


# ----------------------------------------------------------------------------
# Genetik Algoritma
# ----------------------------------------------------------------------------
class GeneticImage:
    """
    Kromozom  : H x W x 3 piksel dizisi (her piksel kanalı bir gen, 0-255 arası float)
    Fitness   : 1 - MSE / 255^2   (1.0 = hedefle birebir aynı)
    Seçilim   : Turnuva seçilimi
    Çaprazlama: Blok bazlı uniform çaprazlama (komşu pikseller birlikte aktarılır)
    Mutasyon  : Genlerin küçük bir kısmına Gauss gürültüsü + nadiren rastgele değer
    Elitizm   : En iyi bireyler doğrudan bir sonraki nesle geçer
    """

    def __init__(self, target: np.ndarray, pop_size=60, elite=4, tournament=3,
                 mutation_rate=0.01, mutation_sigma=40.0, block=4, seed=None):
        self.rng = np.random.default_rng(seed)
        self.target = target.astype(np.float32)
        self.shape = target.shape
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.block = block

        # Başlangıç popülasyonu: tamamen rastgele pikseller
        self.population = self.rng.uniform(0, 255, size=(pop_size, *self.shape)).astype(np.float32)
        self.fitness = self.evaluate(self.population)

    # --- Fitness ---
    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        mse = ((pop - self.target) ** 2).mean(axis=(1, 2, 3))
        return 1.0 - mse / (255.0 ** 2)

    # --- Seçilim ---
    def tournament_select(self, n: int) -> np.ndarray:
        candidates = self.rng.integers(0, self.pop_size, size=(n, self.tournament))
        winners = candidates[np.arange(n), self.fitness[candidates].argmax(axis=1)]
        return self.population[winners]

    # --- Çaprazlama ---
    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        n, h, w, _ = p1.shape
        b = self.block
        bh, bw = -(-h // b), -(-w // b)  # yukarı yuvarlama
        mask = self.rng.random((n, bh, bw, 1)) < 0.5
        mask = mask.repeat(b, axis=1).repeat(b, axis=2)[:, :h, :w]
        return np.where(mask, p1, p2)

    # --- Mutasyon ---
    def mutate(self, pop: np.ndarray, rate: float, sigma: float) -> np.ndarray:
        flat = pop.reshape(-1)
        # Sadece mutasyona uğrayacak genler için rastgele sayı üret (hızlı)
        k = self.rng.binomial(flat.size, rate)
        idx = self.rng.integers(0, flat.size, size=k)
        flat[idx] += self.rng.normal(0, sigma, size=k).astype(np.float32)
        # Çok nadir "büyük sıçrama" mutasyonu: geni tamamen rastgele bir değerle değiştir
        k2 = self.rng.binomial(flat.size, rate * 0.02)
        idx2 = self.rng.integers(0, flat.size, size=k2)
        flat[idx2] = self.rng.uniform(0, 255, size=k2).astype(np.float32)
        np.clip(flat, 0, 255, out=flat)
        return pop

    # --- Bir nesil ---
    def step(self, progress: float):
        # Mutasyon oranı ve şiddeti zamanla azalır: başta kaba arama, sonda ince ayar
        rate = self.mutation_rate * (1.0 - 0.8 * progress)
        sigma = self.mutation_sigma * (1.0 - 0.75 * progress)

        order = np.argsort(self.fitness)[::-1]
        elites = self.population[order[:self.elite]]

        n_child = self.pop_size - self.elite
        p1 = self.tournament_select(n_child)
        p2 = self.tournament_select(n_child)
        children = self.mutate(self.crossover(p1, p2), rate, sigma)

        self.population = np.concatenate([elites, children], axis=0)
        self.fitness = self.evaluate(self.population)

    def best(self) -> tuple[np.ndarray, float]:
        i = int(self.fitness.argmax())
        return self.population[i], float(self.fitness[i])


# ----------------------------------------------------------------------------
# Kaydetme yardımcıları
# ----------------------------------------------------------------------------
def to_image(arr: np.ndarray, display_size: int) -> Image.Image:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.resize((display_size, display_size), Image.NEAREST)


def make_montage(snapshots, target_img: Image.Image, path: str, cell=160, cols=6):
    items = snapshots + [("Hedef", None, target_img)]
    rows = -(-len(items) // cols)
    label_h = 20
    canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (gen, fit, img) in enumerate(items):
        x, y = (k % cols) * cell, (k // cols) * (cell + label_h)
        canvas.paste(img.resize((cell, cell), Image.NEAREST), (x, y))
        text = gen if fit is None else f"Nesil {gen}  ({fit * 100:.1f}%)"
        draw.text((x + 4, y + cell + 3), text, fill="black")
    canvas.save(path)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Genetik algoritma ile yapay yüz görüntüsü üretimi")
    ap.add_argument("--source", choices=["auto", "dataset", "image"], default="auto")
    ap.add_argument("--index", type=int, default=0, help="Dataset'ten kaçıncı görsel")
    ap.add_argument("--image", default="image.jpg", help="Yedek / doğrudan kullanılacak görsel")
    ap.add_argument("--size", type=int, default=64, help="GA'nın çalıştığı çözünürlük (size x size)")
    ap.add_argument("--pop", type=int, default=60, help="Popülasyon büyüklüğü")
    ap.add_argument("--generations", type=int, default=15000)
    ap.add_argument("--save-every", type=int, default=1000, help="Kaç nesilde bir görüntü kaydedilsin")
    ap.add_argument("--mutation-rate", type=float, default=0.01)
    ap.add_argument("--mutation-sigma", type=float, default=40.0)
    ap.add_argument("--block", type=int, default=4, help="Çaprazlama blok boyutu (piksel)")
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ciktilar")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Hedef
    target_full, src_name = load_target(args.source, args.index, args.image)
    target_small = target_full.resize((args.size, args.size), Image.LANCZOS)
    target_arr = np.asarray(target_small, dtype=np.float32)
    display = 256
    target_disp = target_small.resize((display, display), Image.NEAREST)
    target_disp.save(os.path.join(args.out, "hedef.png"))
    print(f"Hedef: {src_name}  (orijinal {target_full.size}, GA çözünürlüğü {args.size}x{args.size})")

    ga = GeneticImage(target_arr, pop_size=args.pop, elite=args.elite,
                      mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma,
                      block=args.block, seed=args.seed)

    snapshots = []

    def save(gen):
        arr, fit = ga.best()
        img = to_image(arr, display)
        img.save(os.path.join(args.out, f"nesil_{gen:06d}.png"))
        snapshots.append((gen, fit, img))
        print(f"  Nesil {gen:6d} | fitness {fit * 100:6.2f}% | kaydedildi")

    save(0)  # tamamen rastgele başlangıç
    t0 = time.time()
    for gen in range(1, args.generations + 1):
        ga.step(gen / args.generations)
        if gen % args.save_every == 0 or gen == args.generations:
            save(gen)
    print(f"Süre: {time.time() - t0:.1f} sn")

    # İlerleme kolajı ve GIF
    make_montage(snapshots, target_disp, os.path.join(args.out, "ilerleme.png"))
    frames = [img for _, _, img in snapshots]
    frames[0].save(os.path.join(args.out, "evrim.gif"), save_all=True,
                   append_images=frames[1:] + [target_disp], duration=400, loop=0)
    print(f"Çıktılar: {os.path.abspath(args.out)}  (ilerleme.png, evrim.gif, nesil_*.png)")


if __name__ == "__main__":
    main()
