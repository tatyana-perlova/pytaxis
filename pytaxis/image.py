#!======Open librairies================
import cv2
import numpy as np
from pandas import DataFrame, Series  # for convenience
import matplotlib.pyplot as plt

#!==============Define functions================
#!=============Convert the image to binary with approproate filtering===========
def get_binary(img):
    '''
    Convert background subtracted image to binary
    Takes:
    img - background subtracted image
    Returns:
    frame_bin - binary image
    '''
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))#create elliptical kernel
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    frame_dilated = cv2.dilate(frame_gray,kernel_close,iterations = 1)
    ret, frame_bin = cv2.threshold(frame_dilated, 0, 255, cv2.THRESH_BINARY_INV)
    
    return(frame_bin)

#!===================Find objects in the binary image======================
def find_cells(frame, background, mnsize, mxsize):
    '''
    Finds contours in the phase contrast image and fits them with ellipses, from which coordinates, angles and lengths are extracted.
    Takes:
    frame - original movie frame
    bckgrnd - background frame
    mnsize - minimum number of pixels in a feature
    mxsize - maximum number of pixels in a feature
    Returns:
    x_coords - list with x coordinates of features in pixels
    y_coords - list with y coordinates of features in pixels
    lengths - list of lengths calculated from major axis
    angles - list of angles
    contours
    img_ori
    '''
    lengths = []
    x_coords = []
    y_coords = []
    angles = []
    
    
    frame_adj = cv2.subtract(background, frame)
    frame_bin = get_binary(frame_adj)
    _, contours, _ = cv2.findContours(frame_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        if (cv2.contourArea(cnt) >= mnsize) & (cv2.contourArea(cnt) <= mxsize):
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            if (x > 0)*(y > 0)*(x<=imdim)*(y <= imdim) > 0:
                cv2.ellipse(frame, (int((x)),int((y))), (int(MA/2 + 3), int(ma/2 + 3)), round(angle), 0, 360, 255, 2)
                x_coords.append(x), y_coords.append(y), lengths.append(MA), angles.append(angle)
                        
    return(x_coords, y_coords, lengths, angles, contours, frame)
    
def get_background(filename, begin = 0, Nframes = 300, imdim = 2048, alpha = 0.005):
    '''
    Gets background from the video before and after the illumination is changed.
    Takes:
    filename - name of the video to be processed.
    begin - first frame from which background accumulation is started
    Nframes - number of frames over which to accumulate background
    imdim - size of the videoframe in pixels
    alpha - weight of the individual frame.
    Returns:
    background - intensity scaled background frame.
    '''
    cap = cv2.VideoCapture(filename)
    background = np.float32(np.zeros((imdim, imdim, 3)))
    
    for i in range(begin):
        ret, frame = cap.read()
    
    for i in range(begin, begin + Nframes):
        ret, frame = cap.read()
        cv2.accumulateWeighted(frame, background, alpha)
        
    cap.release()
    background = cv2.convertScaleAbs(background)
    
    return(background)

def find_cells_video(filename,
                     background,
                     minframe = 0,
                     maxframe = None,
                     imdim = 2048, 
                     mnsize = 15, 
                     mxsize = 150,
                     write = False):
    '''
    Find bacteria in the video before and after the illumination is changed.
    Takes:
    filename - name of the video to be processed.
    bckgrnd - intensity scaled background frame.
    maxframe - frame till which the movie is processed.
    minframe - starting frame.
    imdim - size of the videoframe in pixels.
    mnsize - minimal size of bacteria in pixels.
    mxsize - maximal size of bacteria in pixels.
    write - whether or not a movie with tracked bacteria is recorded or not
    Returns:
    f2 - dataframe with found features and their coordinates
    frame - frame with highlighted bacteria for testing purposes.
    
    '''
    cap = cv2.VideoCapture(filename)
    if write == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename[:-4] + '_detected.avi',fourcc, 6, (imdim, imdim))

    if maxframe == None:
        maxframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1        
    
    for i in range(minframe):
        _, _ = cap.read()
    
    data = {'x': [], 'y': [], 'frame': [], 'body_angle': [], 'length': []}
    for i in range(minframe, maxframe):
        ret, frame = cap.read()
        
        (x_coords, y_coords, lengths, angles, contours, frame) = find_cells(frame, background, mnsize, mxsize)
        if write == True:
            out.write(frame)
        
        if i % 100 == 0:
            print('{} bacteria found in {}th frame'.format(len(x_coords),  i))
            
        data['x'] = data['x'] + x_coords
        data['y'] = data['y'] + y_coords
        data['body_angle'] = data['body_angle'] + angles
        data['length'] = data['length'] + lengths
        data['frame'] = data['frame'] + [i for j in range(len(x_coords))]
    
    cap.release()
    coords = DataFrame(data)
    
    plot_N_bact(coords, frame, filename)
    
    return(coords, frame)

        
def test_detection(filename, mnsize, mxsize, begin = 0, Nframes = 300, imdim = 2048, alpha = 0.005, show = False):
    
    '''
    Accumulates background and finds bacteria in one frame, shows background frame and frame with detected bacteria. For testing purposes.
    Takes:
    filename - name of the video to be processed.
    mnsize - minimal size of bacteria in pixels.
    mxsize - maximal size of bacteria in pixels.
    begin - first frame from which background accumulation is started
    Nframes - number of frames over which to accumulate background
    imdim - size of the videoframe in pixels
    alpha - weight of the individual frame.
    Returns:
    background - background frame
    frame - frame with highlighted bacteria for testing purposes.
    
    '''    
    background = get_background(filename, begin, Nframes, imdim, alpha)
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    (x_coords, _, _, _, _, frame) = find_cells(frame, background, mnsize, mxsize)
    
    if show == True:
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.title('Background frame')
        plt.imshow(background)
        plt.subplot(122)
        plt.title('{} bacteria found'.format(len(x_coords)))
        plt.imshow(frame)
    return(background, frame)
    
def plot_N_bact(coords, frame, filename, save = True):
    
    '''
    Plot number of bacteria in each frame along with one frame with bacteria outlines.
    Takes:
    coords - dataframe with bacterial coordinates in each frame,
    frame - frame with bacteria outlied,
    filename - name of the movie.
    '''
    n_bacteria = coords[['x', 'frame']].groupby(['frame'], as_index = False).count().x
    
    plt.figure(figsize = (18, 5))
    plt.suptitle('Avrerage number of bacteria in {} is {}'.format(filename, int(n_bacteria.mean())))
    
    plt.subplot2grid((1,3), (0,0), colspan=2)
    plt.xlabel('Frame')
    plt.ylabel('Number of detected contours')
    
    plt.plot(n_bacteria)
    plt.subplot(133)
    plt.imshow(frame)
    plt.xticks([])
    plt.yticks([])