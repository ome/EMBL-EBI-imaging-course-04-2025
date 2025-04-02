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

def main():
    cluster = LocalCluster()
    # read the image as dask array
    # Code tests on a 5D-image
    path_to_image = "/path_to_image"
    img = BioImage(path_to_image)
    print(img.dask_data.shape)
    with Client(cluster) as client: # This ensure that we close at the end
        # perform code
        futures = run_analysis(client, img.dask_data)
        results = client.gather(futures)
    print(results)

if __name__ == "__main__":
    main()
