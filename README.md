## Advanced Lane Finding Project

---

**Overview**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chess.png "Chess"
[image2]: ./output_images/combined_binary.png "Combined Binary"
[image3]: ./output_images/perspective.png "Perspective Transformation"
[image4]: ./output_images/findpixel1.png "Find Pixel"
[image5]: ./output_images/radiusFormula.png "Curvature"
[image6]: ./output_images/result.png "Result"
[video1]: ./project_video.mp4 "Video"


---

### Camera Calibration


The code for this step is located in `./camCalibrate.py`. There is one function called `camCalibrate()`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]



## Pipeline

### Color Transforms and Gradients

The code for this step is located in `./thresholder.py`.  In this code there are 4 different funtions (`abs_sobel_thresh , mag_threshold, dir_threshold, hls_threshold`), for generate a binary image with different methods. And one function for combining these methods and produce a result image. Here's an example of my output for this step.

![alt text][image2]


### Perspective Transformation

The code for my perspective transform includes a function called `warper()`, which appears in the file `./@@@@@/example.py`.  The `warper()` function takes as inputs an image (`img`). I defining the source (`src`) and destination (`dst`) points in the funtion.  I chose the hardcode the source and destination points in the following manner:

```python
    
    src = np.array([(580, 460),
                    (205, 720),
                    (1110, 720),
                    (703, 460)],np.float32)
    
    dst = np.array([(320, 0),
                    (320, 720),
                    (960, 720),
                    (960, 0)],np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 205, 720      | 320, 720      |
| 1110, 720     | 960, 720      |
| 703, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

### Identify Lane-Line Pixels and Fit Polynomial

In this step to identifying the position of pixels I used 2 methods, `sliding_window` and `search_around_poly` .

sliding_window simply works like that:

1. Sum the bottom half of the binary image and find pick points
2. By using these pick point define the left and right base points.(Starting point for window)
3. Find non-zero pixels in image by  `nonzero = binary_warped.nonzero()`
4. Go throuh over image from bottom top with windows.
5. While sliding the window, check the position of pick point and rearrange the position of center.
( if number of the pixels in the windows bigger than threshold(minpix) than we set new center position as their mean position.)
        ```python
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        ```
6. Store these indexes into the list.
7. Concatenate these pixels after sliding and get pixel the pixels of left and right with these index.

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]


search_around_poly is a little bit different. The differences is you don't have to look all over image because you have a line from previous frame and you can just look at it's around.

After getting the pixel indexes, we can find a polynomial which fitting with our points. To do that I used `fit_polynomial()` function. It is taking image shape and the pixel indexes. To find a polynomial I used these code:

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

![alt text][image4]


### Curvature and Center Position

The radius of curvature at any point x of the function x = f(y) is given as follows:

![alt text][image5]


I did this in my code by `@@@@@/measure_curvature_real()` function. This function just implement the formula:
        
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
For center position i just finding the midpoint of the left-right line and take differences between image center:

    midpoint = np.int(undistorted.shape[1]/2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

### Sanity Check

Before drawing the lines, I'm checking these:

* Checking that they have similar curvature.

        diff_cr = np.absolute(right_km_cr-left_km_cr)
        if diff_cr>thresh1:
            print("FAİLED - no similir left-right  left: %f  right: %f  diff:  %f"%(left_km_cr,right_km_cr,diff_cr))
            return False

* Checking the new curvature is similar to the average_curv.


        weights = np.arange(1,len(l_radius_list)+1)/len(l_radius_list)
        left_avg =np.average(l_radius_list,0,weights[-len(l_radius_list):])/1000
        right_avg =np.average(r_radius_list,0,weights[-len(l_radius_list):])/1000

        left_diff = np.absolute(left_km_cr-left_avg)
        right_diff = np.absolute(right_km_cr-right_avg)
        
        # also check the last and new curv differences.
        last_diff_l = np.absolute((l_radius_list[-1]/1000)-left_km_cr)
        last_diff_r = np.absolute((r_radius_list[-1]/1000)-right_km_cr)
        
        if left_diff>thresh2 or right_diff>thresh2 or last_diff_l>thresh2 or last_diff_r>thresh2:
            return False


* Checking that they are separated by approximately the right distance horizontally. (max:905 - min:123 distance between left-right)

        diff_list = (right_points-left_points)
        check = ((diff_list>905) | (diff_list<123))
        outlier = diff_list[check]
        if len(outlier)>0:
            print("FAİLED - not parallel / num of outliar: %d"%(len(outlier)))
            return False


### Draw Lane - Result

Here is an example of my result on a test image:

![alt text][image6]

---


### Video Results


Here's the project_video result [link to my video result](./project_video-result.mp4)

Here's the challenge_video result [link to my video result](./challenge_video-result.mp4)

---


 
