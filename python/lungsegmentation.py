from dicom import DicomDataset, DicomViewer
from skimage import measure
import numpy as np

#-------------------------------------------------------------------------------
# Segmentation:
def get_max_label(image, bg_id=-1):
    # Histogram bin counts of the image to identify the largest region of the 
    # image (air outside the body).
    values, counts = np.unique(image, return_counts=True)

    # Remove background counts and label:
    counts = counts[values != bg_id]
    values = values[values != bg_id]

    if len(counts) > 0:
        return values[np.argmax(counts)]
    else:
        return None

def segment(image, include_lung=True):
    # Threshold the HU image to obtain a mask of the image containing only air 
    # and lung. 1 and 2 are the labels assigned and 0 is background.
    mask   = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(mask)
    
    # Background (air around the person) is labeled as background. Assign 
    # appropriate labels:
    mask[labels == labels[0,0,0]] = 2
    
    # Segment the lung area if specified:
    if include_lung:
        # For every slice we determine the largest solid structure
        for ind, slice in enumerate(mask):
            slice -= 1
            slice_label = measure.label(slice)
            lmax = get_max_label(slice_label)
            
            if lmax is not None:
                mask[ind][(slice_label != lmax) & (slice_label != 0)] = 1

    # Invert to make lung the foreground label:
    mask = 2-mask
    
    # Remove other air pockets insided body
    labels = measure.label(mask, background=0)
    lmax = get_max_label(labels, bg_id=0)

    if lmax is not None:
        mask[labels != lmax] = 0
 
    return mask

#-------------------------------------------------------------------------------
# Read and pre-process image:
ds = DicomDataset("./data/lung-ct-dicom/PAT001")
ds.read()
ds.raw2hu()
ds.isotropic_resample(scale=[3,1,1])

# Now, segment the image in the dataset:
lung_mask = segment(ds.image)

# View slices with labels overlayed:
viewer = DicomViewer()
viewer.plot(ds, lung_mask)