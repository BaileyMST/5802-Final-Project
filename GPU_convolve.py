import os
import site
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle

import cupy as cp
import cupyx.scipy.signal as cpsig
import cupyx.scipy.ndimage as cpxndi


# Uncomment the following block if you encounter DLL load issues on Windows for CuPy
# for sp in site.getsitepackages():
#     cand = os.path.join(sp, "cupy", ".data")
#     if os.path.isdir(cand):
#         # Walk to find a directory containing cublas*.dll
#         for root, dirs, files in os.walk(cand):
#             if any(f.lower().startswith("cublas") and f.lower().endswith(".dll") for f in files):
#                 os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
#                 break


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = Image.open(filepath)
    return (np.asarray(img).astype(float)/255)[:, :, :3]


def get_circ_image(image_size, radius):
    """Create an image of width `image_size` with a circle 
    of radius `radius` in its center."""
    assert(image_size % 2 == 1)
    h = (image_size + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Threshold by squared radius
    d_sq = (d_sq <= radius**2).astype(float)
    return d_sq


def plot_circ_features(image, features, ax):
    ax.imshow(image)
    for m in features:
        if len(m) == 0:
            continue
        x, y, sigma = m[0], m[1], m[2]
        radius = sigma*np.sqrt(2)
        cir = Circle((x, y), radius, color='r', fill=False, linewidth=1.5)
        ax.add_artist(cir)


def get_LoG_filter_gpu(kernel_size: int, sigma: float) -> cp.ndarray:
    """
    GPU LoG kernel generator matching your CPU get_LoG_filter() math.
    Returns cp.float32 kernel of shape (ks, ks).
    """
    ks = int(kernel_size)
    sig = float(sigma)
    c = ks // 2

    x = cp.arange(ks, dtype=cp.float32) - cp.float32(c)
    y = cp.arange(ks, dtype=cp.float32) - cp.float32(c)
    xx, yy = cp.meshgrid(x, y, indexing="ij")

    sigma2 = cp.float32(sig * sig)
    sigma4 = sigma2 * sigma2

    front = cp.float32(1.0) / (cp.float32(np.pi) * sigma4)
    mid = cp.float32(1.0) - (xx * xx + yy * yy) / (cp.float32(2.0) * sigma2)
    back = cp.exp(-(xx * xx + yy * yy) / (cp.float32(2.0) * sigma2))

    kernel = front * mid * back
    kernel = kernel * sigma2  # matches: kernel = kernel * sigma**2
    return kernel.astype(cp.float32)


def local_maxima_3d_gpu(response_gpu: cp.ndarray, threshold: float, sigmas_cpu: np.ndarray, neighborhood_size: int = 5):
    data_region_max = cpxndi.maximum_filter(response_gpu, size=neighborhood_size)
    maxima = (response_gpu == data_region_max)

    maxima &= (response_gpu >= cp.float32(threshold))

    coords = cp.argwhere(maxima)
    coords_cpu = coords.get()

    features = []
    for y, x, z in coords_cpu:
        features.append((int(x), int(y), float(sigmas_cpu[int(z)])))
    return features


def compute_multi_scale_features_gpu(image_2d_cpu: np.ndarray, sigmas, threshold: float, window_size: int = 11):
    if image_2d_cpu.ndim != 2:
        raise ValueError("compute_multi_scale_features_gpu expects a single-channel 2D image.")

    sigmas_cpu = np.asarray(sigmas, dtype=np.float32)
    H, W = image_2d_cpu.shape
    S = sigmas_cpu.size

    img_gpu = cp.asarray(image_2d_cpu, dtype=cp.float32)
    response_gpu = cp.empty((H, W, S), dtype=cp.float32)

    for i, sigma in enumerate(sigmas_cpu):
        ksize = int(6.0 * float(sigma)) + 1
        if ksize % 2 == 0:
            ksize += 1

        LoG = get_LoG_filter_gpu(ksize, float(sigma))

        conv = cpsig.convolve2d(
            img_gpu, LoG,
            mode="same",
            boundary="fill",
            fillvalue=0
        )

        response_gpu[:, :, i] = cp.abs(conv)

    features = local_maxima_3d_gpu(
        response_gpu,
        threshold=threshold,
        sigmas_cpu=sigmas_cpu,
        neighborhood_size=window_size
    )
    return features


def cuda_barrier():
    cp.cuda.Stream.null.synchronize()


def cuda_time_block(fn, *args, **kwargs):
    cuda_barrier()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cuda_barrier()
    t1 = time.perf_counter()
    return out, (t1 - t0)


def main():
    os.makedirs("Outputs", exist_ok=True)

    cuda_barrier()
    program_start = time.perf_counter()

    print("CUDA initialized (CuPy).")
    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'])

    # img 0
    im_half_size = 25
    sigmas = np.arange(2, 20, 0.1, dtype=np.float32)

    circ_img_a = get_circ_image(2 * im_half_size + 1, radius=12)
    circ_img_b = -get_circ_image(2 * im_half_size + 1, radius=8)
    circ_img = np.concatenate([circ_img_a, circ_img_b], axis=1)

    plt.figure(figsize=(8, 4))
    plot_circ_features(circ_img, [], plt.gca())
    plt.title("Input (synthetic)")
    plt.savefig("Outputs/input_circle.png", dpi=300)

    # Compute-only timing (circle)
    features, compute_time = cuda_time_block(
        compute_multi_scale_features_gpu,
        circ_img, sigmas,
        threshold=0.01,
        window_size=11
    )

    plt.figure(figsize=(8, 4))
    plot_circ_features(circ_img, features, plt.gca())
    plt.title("Detected Features (GPU)")
    plt.savefig("Outputs/detected_features_circle.png", dpi=300)
    print("Saved detected features for circle image.")
    print(f"Compute-only time (circle, CUDA): {compute_time:.6f} s")

    # img 1
    img1 = load_image("Inputs/sunflower_field.jpg")[:, :, 0]
    sigmas2 = np.arange(4, 80, 2, dtype=np.float32)

    features, compute_time = cuda_time_block(
        compute_multi_scale_features_gpu,
        img1, sigmas2,
        threshold=0.05,
        window_size=11
    )

    plt.figure(figsize=(8, 8))
    plot_circ_features(img1, features, plt.gca())
    plt.title("Detected Features (Sunflower, GPU)")
    plt.savefig("Outputs/detected_features_sunflower.png", dpi=300)
    print("Saved detected features for sunflower image.")
    print(f"Compute-only time (sunflower, CUDA): {compute_time:.6f} s")

    # img 2
    img2 = load_image("Inputs/hornet.jpg")[:, :, 0]

    features, compute_time = cuda_time_block(
        compute_multi_scale_features_gpu,
        img2, sigmas2,
        threshold=0.05,
        window_size=11
    )

    plt.figure(figsize=(8, 8))
    plot_circ_features(img2, features, plt.gca())
    plt.title("Detected Features (Hornet, GPU)")
    plt.savefig("Outputs/detected_features_hornet.png", dpi=300)
    print("Saved detected features for hornet image.")
    print(f"Compute-only time (hornet, CUDA): {compute_time:.6f} s")

    # "program_end" equivalent (end-to-end runtime)
    cuda_barrier()
    program_end = time.perf_counter()
    total_time = program_end - program_start
    print(f"End-to-end CUDA runtime: {total_time:.6f} seconds")


if __name__ == "__main__":
    main()
