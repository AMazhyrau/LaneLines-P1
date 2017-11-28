# Finding Lane Lines on the Road

### Pipeline description

1. Normalize brightness with CLAHE (Contrast Limited Adaptive Histogram Equalization)
   to improve the contrast of the image and apply gaussian blur filter to reduce image noise

~~~python
def normalize_image(lab_image, clip_limit=3.0, tile_grid_size=(8, 8)):
    l, a, b = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
~~~

2. Create vieport(select region of interest) and apply color select

~~~python
def crop_image(rgb_image):
    height, width = rgb_image.shape[:2]
    bottom_left = [0, height]
    bottom_right = [width, height]
    top_left = [width / 2 - 50, height / 2 + 60]
    top_right = [width / 2 + 50, height / 2 + 60]
    viewport_params = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cropped_image = region_of_interest(rgb_image, viewport_params)

    return cropped_image
   
~~~

3. Apply color selection

~~~python
color_thresholds = (image[:, :, 0] < 200) | (image[:, :, 1] < 180) | (image[:, :, 2] < 0)
color_select = cropped_image.copy()
color_select[color_thresholds] = [0, 0, 0]
~~~

4. Convert the image to gray color

~~~python
cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
~~~

5. Apply Canny edge detection with cv2.dilate method. It increases the white region in the image, i.e. makes lines more 
solid

~~~python
edges = canny(gray_image, 50, 150)
edges = cv2.dilate(edges, kernel=(5,5), iterations=1)
~~~

6. Apply Hough transform `cv2.HoughLinesP` to detect lane lines.

~~~python
hough_lines = cv2.HoughLinesP(
    edges, 
    rho=1, 
    theta=np.pi / 180, 
    threshold=35, 
    minLineLength=10, 
    maxLineGap=10
)
~~~

There are tuning params:

- rho: distance resolution of the accumulator in pixels.
- theta: angle resolution of the accumulator in radians.
- threshold: accumulator threshold parameter. Only those lines are returned that get enough votes (greater than `threshold`).
- minLineLength: minimum line length. Line segments shorter than that are rejected.
- maxLineGap: maximum allowed gap between line segments to treat them as single line.

7. Average/extrapolate lane lines and draw avg lines

- Extract line coordinates `x1, y1, x2, y2`
- Ignore all vertical lines
- Compute slope factor `(y2 - y1) / (x2 - x1)` and ignore almost horizontal lines, where `abs(slope) < 0.4`.
  Select left and right lines by slope factor.
- Extract abscissa and ordinate for left and right lines, then apply least squares polynomial 
  fit with `numpy.polyfit(x, y, deg = 1)`
- Convert a line represented in slope and intercept into pixel points

~~~python
def get_points_by_slope(image, line):
    if line is None:
        return None

    slope, intercept = line

    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 0.62)  # slightly lower than the middle, kludge to prevent line crossing

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return (x1, y1), (x2, y2)
~~~

- Average lines with `numpy.mean(lines, axis=0, dtype=np.int32)`
- Draw lines

~~~python
def draw_lane_lines(image, lines, color=(255, 0, 0), thickness=10):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 0.7, line_image, 1.0, 0.0)
~~~

### 2. Potential shortcomings with my current pipeline

I still have a challenges to solve such as:
- different type of road material;
- different brightness, I tried to solve this with CLAHE algorithm;

I have no idea what will happen at crossroads or in road repairing case, i.e. how to detect lane lines in such cases?
I think this pipeline will not work at night, or in bad weather when we have bad lighting with gradient

### 3. Possible improvements to my pipeline

- Add camera calibration.
- Implement method that will calculate the distance between lines, to find the right group of lines.
- Determine the curvature of the line to understand where the road turns.
- Try to find more effective algorithm to normalize brightness.
