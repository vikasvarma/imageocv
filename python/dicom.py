import numpy as np
import pydicom
import os
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

from skimage import measure, color, img_as_uint
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#-------------------------------------------------------------------------------
class DicomDataset:

    def __init__(self, path):
        """ Constructor - Scans the input path for dicom dataset and caches     
                          all available dicom files.
        """

        self.slices  = None
        self.image   = None
        self.spacing = None

        # Scan for all dicom files in the path provided:
        if os.path.isdir(path):
            self.files = [os.path.join(path, f) for f in os.listdir(path)]
        elif os.path.splittext(path)[1] == "dcm":
            self.files = path
        else:
            raise Exception("Invalid path specified.")

        if self.files is None:
            raise Exception("No dicom files found in the path provided.")

    def read(self):
        """
            Read all the files sequentially and construct the image/volume based on slice positions.
        """

        self.slices = [pydicom.read_file(dcm) for dcm in self.files]
        self.slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        # Estimate slice thickness in Z direction
        try:
            thickness = np.abs(self.slices[0].ImagePositionPatient[2] - 
                               self.slices[1].ImagePositionPatient[2])
        except:
            thickness = np.abs(self.slices[0].SliceLocation - 
                               self.slices[1].SliceLocation)

        for slice in self.slices:
            slice.SliceThickness = thickness

        # Construct the image:
        self.image = np.stack([slice.pixel_array for slice in self.slices])

        # Calculate pixel spacing:
        self.spacing = np.append(np.float32(self.slices[0].SliceThickness), 
                                np.float32(self.slices[0].PixelSpacing))

    def raw2hu(self):
        """
            Convert RAW dicom pixels to Hounsfield Unit (HU) - a measure of radiodensity.
        """

        hu = self.image.astype(np.int16)

        # Set pixels that fall outside the scan bounds to zero intensity:
        hu[hu <= -1000] = 0

        # Now, translate the slice intensities to HU:
        for slice_num in range(len(self.slices)):
            # Read intercept and slope from metadata:
            intercept = self.slices[slice_num].RescaleIntercept
            slope = self.slices[slice_num].RescaleSlope
            hu[slice_num] = (slope * hu[slice_num].astype(np.float64)).astype(np.int16) + np.int16(intercept)

        self.image = np.array(hu, dtype=np.int16)
    
    def isotropic_resample(self, scale=[1,1,1]):
        """
            Perform isotropic resampling to resolve inconsistent pixel spacing in the slice dimension. Input image is a stack of 
        """

        target = np.round(self.image.shape * (self.spacing / scale))
        factor = target / self.image.shape
        
        # Update image and set the new pixel spacing:
        self.spacing = self.spacing / factor
        self.image = scipy.ndimage.interpolation.zoom(
            self.image, factor, 
            mode='nearest'
        )

    def plot(
        self,
        threshold = -300,
        size = (10,10),
        face_color = [0.45, 0.45, 0.75],
        alpha = 0.7
        ):
        """
            Plot the dataset on a 3D axes. This is extremely slow and resource heavy. Instead use the slice viewer.
        """
        im = self.image.transpose(2,1,0)
        
        verts, faces = measure.marching_cubes_classic(im, threshold)

        fig = plt.figure(figsize=size)
        ax  = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, im.shape[0])
        ax.set_ylim(0, im.shape[1])
        ax.set_zlim(0, im.shape[2])

        plt.show()

#-------------------------------------------------------------------------------
class DicomViewer():
    def __init__(self, mode='xy'):
        self.mode = 'xy'
        
        self.__figure__   = None
        self.__axes__     = None
        self.__dataset__  = None
        self.__slicenum__ = None
        self.__image__    = None
        self.__labels__   = None

    def plot(self, dataset, labels = None):
        
        # Determine if the input dataset is valid:
        if not isinstance(dataset, DicomDataset):
            raise Exception("Input must be a DicomDataset object.")

        try:
            # Cache the image:
            self.__image__  = dataset.image
            self.__labels__ = labels
        except:
            raise Exception("Unable to read image from the dataset.")

        # Determine if a new figure needs to be created:
        if (self.__figure__ == None or self.__axes__ == None or \
           (self.__figure__ != None and ~plt.fignum_exists(self.__figure__.number))):
            # Axes or figure instance is invalid, create a new instance:
            self.__figure__, self.__axes__ = plt.subplots()
        
        # Open the image with the centre slice:
        self.__slicenum__ = self.__image__.shape[0] // 2
        self.__disp__()

        # Wire up a listener to respond to key presses:
        self.__figure__.canvas.mpl_connect('key_press_event', self.key_press_callback)
        plt.show()

    def __disp__(self):
        # Display the N-th slice of the image:
        index = self.__slicenum__
        slice = self.__image__[index]

        # Convert image to uint8:
        slice = cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX)
        slice = slice.astype(np.uint8)

        if self.__labels__ is not None:
            labels = self.__labels__[index].astype(slice.dtype)
            slice = color.label2rgb(labels, slice, bg_label=0)
        
        if hasattr(self.__axes__, 'images') and len(self.__axes__.images) > 0:
            self.__axes__.images[0].set_array(slice)
        else:
            self.__axes__.imshow(slice)

    def key_press_callback(self, event):
        if event.key == 'n':
            self.next()
        elif event.key == 'p':
            self.prev()
        elif event.key == 'f':
            self.first()
        elif event.key == 'l':
            self.last()

        self.__figure__.canvas.draw()

    def next(self):
        # Go to the next slice:
        if self.__slicenum__ < self.__image__.shape[0]-1:
            self.__slicenum__ += 1
            self.__disp__()
        else:
            pass # Reached the end, no-op.

    def prev(self):
        # Go to the previous slice:
        if self.__slicenum__ > 0:
            self.__slicenum__ -= 1
            self.__disp__()
        else:
            pass # At the first slice, no-op.

    def first(self):
        self.__slicenum__ = 0
        self.__disp__()

    def last(self):
        self.__slicenum__ = self.__image__.shape[0]-1
        self.__disp__()
