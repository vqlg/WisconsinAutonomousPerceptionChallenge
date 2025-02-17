import cv2
import numpy as np

def is_cone_by_slices(contour, min_ratio=1.0):

    # Make a bounding rectangle around cone contour
    x, y, width, height = cv2.boundingRect(contour)
    if width == 0 or height < 2:
        return False

    # Divide bounding rectangle into top & bottom sides
    half_height = height // 2
    top_half_points = []
    bottom_half_points = []

    # Split contour points into top & bottom 
    for point in contour:
        x_point, y_point = point[0]  # Contour points are wrapped in [[x, y]]
        if y_point >= y and y_point < y + half_height:
            top_half_points.append((x_point, y_point))
        else:
            bottom_half_points.append((x_point, y_point))

    # Make sure both sides have points
    if not top_half_points or not bottom_half_points:
        return False

    # Get bounding rectangles for both top & bottom
    _, _, top_width, _ = cv2.boundingRect(np.array(top_half_points))
    _, _, bottom_width, _ = cv2.boundingRect(np.array(bottom_half_points))

    # Make sure the width on bottom > top width by a ratio of 1
    if top_width == 0:
        return False
    width_ratio = bottom_width / float(top_width)
    return width_ratio >= min_ratio

# Load original image
image_path = 'Red.png'
image = cv2.imread(image_path)
if image is None:
    print("Could not load image.")
    exit()

# Change image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create color range to find red items
lower_red1 = np.array([0, 170, 90])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Make binary masks to look for red in both ranges
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
mask = mask_red_combined

# Use morphological operations to decrease noise (get rid of small red items)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Look for contours in processed mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cone_centers = []

# Find dimensions of the image
image_height, image_width = image.shape[:2]

# Calibrate each contour to see if it is a cone
for contour in contours:
    contour_area = cv2.contourArea(contour)
    if contour_area < 200:  # Ignore small contours
        continue

    # Check if contour matches shape of a cone
    if not is_cone_by_slices(contour):
        continue
    
    # Calculate center of the cones contour
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        continue
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    cone_centers.append((center_x, center_y))

# Categorize cones into left and right using how far they are from the center of image
left_cones = [point for point in cone_centers if point[0] < image_width // 2]
right_cones = [point for point in cone_centers if point[0] >= image_width // 2]

def draw_line_polyfit(points, color):
    if len(points) < 2:
        return
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    slope, intercept = np.polyfit(x_coords, y_coords, 1)
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min = int(slope * x_min + intercept)
    y_max = int(slope * x_max + intercept)
    cv2.line(image, (x_min, y_min), (x_max, y_max), color, 3)

# Create lines of best fit for left and right cones
draw_line_polyfit(left_cones, (0, 255, 0))
draw_line_polyfit(right_cones, (0, 255, 0))

# Create final image with lines drawn
cv2.imwrite("answer.png", image)
