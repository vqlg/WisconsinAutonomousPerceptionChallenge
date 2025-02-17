# WisconsinAutonomousPerceptionChallenge
Coding Challenge for WA
![answer](https://github.com/user-attachments/assets/2220377f-6de7-4f10-b584-ce3d14b264b0)

# Methodolgy
I first began by understanding the problem knowing that I had to develop a program that could find the red cones in the image and then create a path based on those cones and I had to use image processing. I realized that I had to do the following: 

-- Find the cones in the image and seperate from other objects
-- Create a line of best fit through the cones to make a path

**After understanding this, I began implementing a solution using computer vision through OpenCV. Here were my steps: **

I detected the color red (using HSV values) and created a range to detect different shades of it. I then used binary masks to make the red color white and then everything else black.

I filtered out noise using morphological operations (to get rid of random small red items)

I detected the cones shape using contours (around the cones edges) from the mask

I had to implement a function to differentiate between cones and other reddish objects like the door and the exit sign. I created a bounding rectangle to check if the base of the cone is wider than its top side (to indicate that its a cone shape) + see if the height is longer than width. 

I had to find the location of each cone. I used cv2.moments() to find the center of the contoured cone.

I separated the left and right cones to create two different lines of best fit. I classified the two sides of the cones using their coordinates relative to the center of the png. 

I used polynomial regression to make a line going through both sides of the cones. 

Finally, I drew the lines onto the original image and created a new image called “answer.png”



# What did you try and why do you think it did not work.

When first trying the color thresholding I had issues finding different HSV values to detect the red cones and it did not work because some of the red shades were not being seen and my range for HSV was way too small. 

I tried filtering via the height on the image by removing any red objects when they appeared in the top ⅓ of the image. This didn't work because cones in the top of the image were not being seen. I fixed it by adding a method that relies on the contour shape of the cone. 

I also had issues with the contour area of the cone, but I just had to find the right area using trial and error. 

I tried cv2.fitLine() for making my line of best fit but it did not generate a correct output line (it didn't go over every line). This was because it was very sensitive to outliers. 




# What libraries are used

**OpenCV (cv2)** -- To process images, detecting HSV values from the cones, contour detection, and drawing the path lines

**NumPy (numpy)** -- To use arrays/change them and for polynomial regression line
