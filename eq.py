__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, gdalnumeric, ogr, osr

from skimage import data, img_as_float
from skimage import exposure
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
# import cv2
# import numpy as np

def histogramcv(path, band):
    img = cv2.imread(path)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # cv2.imshow('Color input image', img)
    # cv2.imshow('Histogram equalized', img_output)
    cv2.imwrite(str(band)+".png",img_output)
    print type(img_)
    return img_output

# cv2.waitKey(0)


# matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def equalizer(path):
    # Load an example image
    img = gdal.Open(path)
    geoTrans = img.GetGeoTransform()
    proj = img.GetProjection()
    imgenhance = gdal.GetDriverByName('GTiff').Create('imgenhance.tif', img.RasterXSize, img.RasterYSize, img.RasterCount,gdal.GDT_UInt16)
    for band in range(img.RasterCount):
        band += 1
        imgaux = img.GetRasterBand(band).ReadAsArray()
        newband = histogram(imgaux,band)
        imgenhance.GetRasterBand(band).WriteArray(newband)
    imgenhance.SetGeoTransform(geoTrans)
    imgenhance.SetProjection(proj)



def histogram(img,band):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2,4), dtype=np.object)
    axes[0,0] = fig.add_subplot(2, 4, 1)
    for i in range(1,4):
        axes[0,i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0,4):
        axes[1,i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.savefig(str(band)+".png")    
    print "continuo"
    return img_eq


equalizer('Testimg/orto/Clip1.tif')




