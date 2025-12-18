import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage
import scipy.signal
from matplotlib.patches import Circle
import mpi4py.MPI as MPI


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


def get_LoG_filter(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i-kernel_size//2
            y = j-kernel_size//2
            front = (1/(np.pi*(sigma**4)))
            mid = (1-(x**2+y**2)/(2*(sigma**2)))
            back = np.exp(-(x**2+y**2)/(2*(sigma**2)))
            kernel[i,j] = front*mid*back
    kernel = kernel*sigma**2
    return kernel


def plot_circ_features(image, features, ax):
    ax.imshow(image)
    for m in features:
        if len(m) == 0:
            continue
        x, y, sigma = m[0], m[1], m[2]
        radius = sigma*np.sqrt(2)
        cir = Circle((x, y), radius, color='r', fill=False, linewidth=1.5)
        ax.add_artist(cir)


def apply_filter(signal, filt):
    """Apply a filter to an image; wrapper around scipy."""
    return scipy.signal.convolve2d(signal, filt, mode='same')

       
def get_local_maxima_3D(data, threshold, sigmas, neighborhood_size=5):
    # See: https://stackoverflow.com/a/9113227/3672986
    data_region_max = scipy.ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_region_max)
    data_min = scipy.ndimage.minimum_filter(data, neighborhood_size)
    maxima[data < threshold] = 0

    labeled, num_objects = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled)

    features = []
    x, y = [], []
    for dy, dx, dz in slices:
        x_center = int(round((dx.start + dx.stop - 1)/2))
        y_center = int(round((dy.start + dy.stop - 1)/2))
        z_center = int(round((dz.start + dz.stop - 1)/2))
        features.append((x_center, y_center, sigmas[z_center]))
    return features


def compute_multi_scale_features(image, sigmas, threshold, window_size=11):
    response = np.zeros((image.shape[0], image.shape[1], sigmas.size))
    num_sigmas = len(sigmas)
    sigma_per_proc = num_sigmas // size
    start_index = rank * sigma_per_proc
    if rank == size - 1:
        end_index = num_sigmas
    else:
        end_index = start_index + sigma_per_proc
    local_sigmas = sigmas[start_index:end_index]
    local = np.zeros((image.shape[0], image.shape[1], len(local_sigmas)))
    for i, sigma in enumerate(local_sigmas):
        LoG_filter = get_LoG_filter(int(6*sigma)+1, sigma)
        feature_response = apply_filter(image, LoG_filter)
        local[:, :, i] = np.abs(feature_response)
    local_full = np.zeros_like(response)
    local_full[:, :, start_index:end_index] = local
    comm.Reduce(local_full, response, op=MPI.SUM, root=0)

    if rank == 0:
        features = get_local_maxima_3D(response, threshold, sigmas, window_size)
        return features
    else:
        return None


if not MPI.Is_initialized():
    MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
program_start = MPI.Wtime()

if rank == 0:
    print(f"MPI initialized with {size} processes.")
im_half_size = 25
fig = plt.figure()
sigmas = np.arange(2, 20, 0.1)
circ_img_a = get_circ_image(2 * im_half_size + 1, radius=12)
circ_img_b = -get_circ_image(2 * im_half_size + 1, radius=8) 
circ_img = np.concatenate([circ_img_a, circ_img_b], axis=1)
plot_circ_features(circ_img, [], plt.gca())
fig = plt.figure(figsize=(8, 8))

comm.Barrier()
t0 = MPI.Wtime()

features = compute_multi_scale_features(circ_img, sigmas, threshold=0.01)

comm.Barrier()
t1 = MPI.Wtime()
compute_time = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

if rank == 0:
    plot_circ_features(circ_img, features, plt.gca())
    plt.title("Detected Features")
    plt.savefig("Outputs/detected_features_circle_mpi.png", dpi=300)
    print("Saved detected features for circle image.")
    print(f"Compute-only time (circle): {compute_time:.6f} s")
img1 = load_image("Inputs/sunflower_field.jpg")[:, :, 0]
fig = plt.figure(figsize=(8, 8))
sigmas = np.arange(4,80,2)

comm.Barrier()
t0 = MPI.Wtime()

features = compute_multi_scale_features(img1, sigmas, threshold=0.05)

comm.Barrier()
t1 = MPI.Wtime()
compute_time = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

if rank == 0:
    plot_circ_features(img1, features, plt.gca())
    plt.title("Detected Features")
    plt.savefig("Outputs/detected_features_sunflower_mpi.png", dpi=300)
    print("Saved detected features for sunflower image.")
    print(f"Compute-only time (circle): {compute_time:.6f} s")
img2 = load_image("Inputs/hornet.jpg")[:, :, 0]
fig = plt.figure(figsize=(8, 8))

comm.Barrier()
t0 = MPI.Wtime()

features = compute_multi_scale_features(img2, sigmas, threshold=0.05)

comm.Barrier()
t1 = MPI.Wtime()
compute_time = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

if rank == 0:
    plot_circ_features(img2, features, plt.gca())
    plt.title("Detected Features")
    plt.savefig("Outputs/detected_features_hornet_mpi.png", dpi=300)
    print("Saved detected features for hornet image.")
    print(f"Compute-only time (circle): {compute_time:.6f} s")

comm.Barrier()
program_end = MPI.Wtime()
elapsed = program_end - program_start

total_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
if rank == 0:
    print(f"End-to-end MPI runtime: {total_time:.6f} seconds")
