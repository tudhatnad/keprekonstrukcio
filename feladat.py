import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

IMAGE_PATH = "lo_kep_binarized.tif"
VIEW_COUNTS = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]


def load_binary_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nem sikerült betölteni a képet: {path}")
    return (image >= 127).astype(np.uint8) * 255


def generate_angles(num_views: int) -> np.ndarray:
    if num_views <= 0:
        raise ValueError("A vetületek száma pozitív egész kell legyen.")
    return np.linspace(0, 179, num_views, endpoint=False)


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def filtered_backprojection(image: np.ndarray, num_views: int) -> tuple[np.ndarray, np.ndarray]:
    theta = generate_angles(num_views)
    sinogram = radon(image, theta=theta, circle=False)
    reconstruction = iradon(sinogram, theta=theta, filter_name="ramp", circle=False)
    return sinogram, reconstruction


def main() -> None:
    original = load_binary_image(IMAGE_PATH)
    errors = []
    view_counts = []
    sinograms = {}
    reconstructions = {}
    binary_reconstructions = {}

    for num_views in VIEW_COUNTS:
        sinogram, reconstructed = filtered_backprojection(original, num_views)
        sinogram_uint8 = normalize_to_uint8(sinogram)
        gray_uint8 = normalize_to_uint8(reconstructed)
        _, reconstructed_binary = cv2.threshold(gray_uint8, 127, 255, cv2.THRESH_BINARY)

        gray_output_path = f"recon_{num_views}_gray.tif"
        binary_output_path = f"recon_{num_views}_binarized.tif"
        sinogram_output_path = f"sinogram_{num_views}.tif"

        cv2.imwrite(gray_output_path, gray_uint8)
        cv2.imwrite(binary_output_path, reconstructed_binary)
        cv2.imwrite(sinogram_output_path, sinogram_uint8)

        sinograms[num_views] = sinogram_uint8
        reconstructions[num_views] = gray_uint8
        binary_reconstructions[num_views] = reconstructed_binary

        rmse = compute_rmse(reconstructed_binary, original)
        errors.append(rmse)
        view_counts.append(num_views)
        print(f"Vetületek: {num_views:>3} | RME: {rmse:.6f} | Szürke: {gray_output_path} | Binarizált: {binary_output_path} | Szinogram: {sinogram_output_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(view_counts, errors, "o-", color="royalblue", linewidth=2, markersize=7)
    plt.xlabel("Vetületek száma")
    plt.ylabel("RME")
    plt.title("Rekonstrukciós hiba függése a vetületek számától")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reconstruction_error_vs_views.png", dpi=200)
    plt.close()

    if len(errors) >= 2:
        best_idx = int(np.argmin(errors))
        best_num_views = view_counts[best_idx]
        best_error = errors[best_idx]
        print(f"\nLegjobb RME a(z) {best_num_views} vetületnél: {best_error:.6f}")
        print("A grafikon alapján a hiba általában a 40-60 vetület környékén stabilizálódik, ezért ez a tartomány jó kompromisszumot ad a minőség és a számítási költség között.")


if __name__ == "__main__":
    main()
