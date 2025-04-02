from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.filters import gaussian
from skimage.measure import label

from dask.distributed import Client, LocalCluster
from bioio import BioImage

def analyze(image_data, x):
    t = 0
    c = 0
    z = 0
    plane = image_data[t, c, z, x:x+50, x:x+50].compute()
    smoothed_image = gaussian(plane, sigma=1)
    black_white_plane = closing(smoothed_image > threshold_otsu(plane))
    label_image = label(black_white_plane)
    name = "x:%s" % (x)
    return label_image, name


def run_analysis(client, image_data):
    futures = []
    values = [50, 100, 150]
    for x in values:
        futures.append(client.submit(analyze, image_data, x))
    return futures

if __name__ == "__main__":
    path_to_image = "/path_to_image"
    img = BioImage(path_to_image)
    print(img.dask_data.shape)
    futures = []
    values = [50, 100, 150]

    client = Client(processes=False, threads_per_worker=1)
    futures = run_analysis(client, img.dask_data)
    results = client.gather(futures)
    print(results)
