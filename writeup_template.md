# **Finding Lane Lines on the Road**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

### Overview

---

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

The goals / steps of this project are the following:

- Make a pipeline that finds lane lines on the road
- Reflect on your work in a written report

---

### Reflection

### 1. Pipeline

My pipeline consist of 7 steps.

```
def myPipeline(img):

    # Create a gray copy of an image
    img_copy = np.copy(img)
    gray = grayscale(img_copy)

    # Blur the gray image for noise reduction
    blur_gray = gaussian_blur(gray, 5)


    # Detect edges using Canny function
    edges = canny(blur_gray, 50, 150)

    # Define region of interest
    imshape = img.shape
    vertices = np.array([[(75, imshape[0]), (420, 325), (510, 325),
                      (915, imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices=vertices)


    # HoughLineP function parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 35     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5  # minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    # Connect detected lines using hough_lines function
    lines = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)

    # Create final image using weighted_img() function on img_copy and
    # lines (result of hough_lines function)
    w_img = weighted_img(lines, img_copy)
    plt.imshow(w_img)
    return w_img

```

Grayscale:

Firstly, image is converted to grayscale...

<img src="examples\myGray.png" width="480" alt="grayscale image" />

Blurred image:

then using gaussian_blur function for gaussian smoothing, and noise reduction.

<img src="examples\myBlureGray.png" width="480" alt="blured image" />

Canny:

To edge detection it is used opencv canny function.

<img src="examples\myEdges.png" width="480" alt="canny image" />

Region of Interest:

Returned edges from previous function are fed into region_of_interest function to filter the region of interest - ROI for futher processing.

<img src="examples\myMaskedImage.png" width="480" alt="region of interest image" />

Hough lines:

To connect lines hough_lines() function is used which uses modified draw_lines() function.
Basically, it is needed to collect slopes and interceptions of each lane line for each frame (if speaking of video), and calculate average of them. To determine which lane line is left or right it is checked if line slope is positive or negative. Using  formula y = mx + b new averaged x-coordinates are found and at the end drawn on starting image

```
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    rlm = [] # right line m (slope)
    rlb = [] # right line b (intercept)
    llm = [] # left line m (slope)
    llb = [] # left line m (intercept)
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1,y1),(x2,y2), color, thickness)
            m = (y2-y1)/(x2-x1)
            b = y1 - m * x1
            # if slope is positive we are considering  right lane line
            if m >= 0.5 and m < 0.8:
                rlm.append(m)
                rlb.append(b)
            # if slope is negative we are considering  left lane line
            elif m<=-0.5 and m > -0.8:
                llm.append(m)
                llb.append(b)

    # Defining fixed y-coordinates           
    y_max = img.shape[0]
    y_min = 325
    
    # if there is slope and intercept, find average
    if rlm  and rlb:
        avg_rlm = float(sum(rlm)/len(rlm))
        avg_rlb = float(sum(rlb)/len(rlb))
        rlx_min = int((y_min-avg_rlb)/(avg_rlm))
        rlx_max = int((y_max-avg_rlb)/(avg_rlm))
        # draw line
        cv2.line(img, (rlx_min,y_min),(rlx_max,y_max), color, thickness=10)
    
    # if there is slope and intercept, find average
    if llm and llb:
        avg_llm = float(sum(llm)/len(llm))
        avg_llb = float(sum(llb)/len(llb))
        llx_min = int((y_min-avg_llb)/(avg_llm))
        llx_max = int((y_max-avg_llb)/(avg_llm))
        cv2.line(img, (llx_min,y_min),(llx_max,y_max), color, thickness=10)

```


<img src="examples\myLines.png" width="480" alt="hough lines image" />

Final - weighted image:

Finally, that region is added on original (copied) image.

<image src="examples\myWimg.png" width="480" alt="final image" />

### 2. Identify potential shortcomings with your current pipeline

1. One potential shortcoming is it only detects the straight lane lines. Challenge task with curved lane lines is not performed well.

2. Potential issue is if vehicle is positioned left or right of the central line. This will probably impact the slope calculation.

### 3. Suggest possible improvements to your pipeline

1. Current algorithm only detects straight lines, so further improvements will be in that direction. Possible solution to that issue, maybe, to segment lines into several parts checking and comparing slopes of previous and next  with the current one.
