from cv2 import cv2
import skimage
from PIL import Image, ImageDraw
import math
import imutils
from imutils import contours
import numpy as np
import sys
import os.path
import time
from tkinter import *
from tkinter import colorchooser
from tkinter.colorchooser import askcolor

'''
This function is used to obtain the edges of the shape.
'''
def pre_process(image, threshold_value_1, threshold_value_2):
    #convert the image to gray to reduce computational complexity as
    #only dealing with one colour channel
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    gray = np.asarray(gray)

    #blur the image, sufficiently enough to remove some of the higher frequency noise, and smooth the edges.
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
 
    ##changing thresholds doesn't affect area proportions but can affect  to fill shape and identify 
    ##edges with lower frequency change
    edges = cv2.Canny(gray, threshold_value_1, threshold_value_2)
    
    #returns the edge image
    return edges






'''
Function used to find the set of contour points which encompass the largest area.
This aims to find the set of contour points which make up the entire exterior of the shape.
It will then return which position in the contour array this set exists at.
'''
def find_biggest_contour(contours):
    highest = 0
    position = 0
    i = 0
    
    #compare each contour in the set to find the set of contour
    #points which encompass the largest area (the outline of the object shape)
    #linear search O(n)
    for c in contours:
        area = cv2.contourArea(c)

        #compare the area of these  contour points to the current highest area.
        if(area > highest):
            highest = area
            position = i
        i += 1
    #return the position where this largest area was found 
    return position





'''
This function finds the centroid of the objects 2d Shape. This is used to then map the image
to find the proportional area of each bin.
'''
def get_object_centroid(image):
    #find the moments of the binary image
    imageMoments = cv2.moments(image)
    x = int(imageMoments["m10"] / imageMoments["m00"])
    y =  int(imageMoments["m01"] / imageMoments["m00"])
    
    #return co ordinates of central point
    return x, y
    
    
    
    

'''
This function is used to turn the edge image into a filled version.
This essentially returns a binary image of the exterior of the shape filled in
meaning any small gaps or holes in the print. 
'''
def fill_image(image):
    filled_image = np.zeros_like(image)
    #finds the largest contour set and then fills between all of those points to 
    #create the binary filled image.
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(filled_image, [contours[0]], 0, 255, -1)
    return filled_image






'''
This function is used to find the total area (white pixel space)
of an image, to then find the percentage area of each bin compared
to the shape area overall
'''
def find_total_area(image):
    pixel_values = np.asarray(image)
    pixel_count = 0
    total_pixels = 0

    for pixel_row in pixel_values:
        for pixel_value in pixel_row:
            #check if the pixel is white to add to the area count
            if(pixel_value == 255):
                pixel_count = pixel_count + 1
            total_pixels = total_pixels + 1
    
    return total_pixels, pixel_count






'''
This function serves to calculate the number of white pixels that exist
in each of the bins. It does this by splitting the shape into 8, and counting the 
white pixels in whichever bin it's currently inside. It returns each bin value in an
array
'''
def find_bin_area(image, cX, cY):
    values = np.asarray(image)
    bin_areas = [0] * 8
    
    row_position = 0
    column_position = 0
    for y in values:
        row_position = 0    
        for x in y:
            if(column_position < cY):
                if(x == 255):
                    if(row_position <= cX/2):
                        bin_areas[0] = bin_areas[0] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[1] = bin_areas[1] + 1.0
                    if(row_position >= cX and row_position <= (cX + (cX/2))):
                        bin_areas[2] = bin_areas[2] + 1.0
                    if(row_position >= cX + (cX/2)):
                        bin_areas[3] = bin_areas[3] + 1.0
            if(column_position > cY):
                if(x == 255):
                    if(row_position <= cX/2):
                        bin_areas[4] = bin_areas[4] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[5] = bin_areas[5] + 1.0
                    if(row_position >= cX and row_position <= (cX + (cX/2))):
                        bin_areas[6] = bin_areas[6] + 1.0
                    if(row_position >= cX + (cX/2)):
                        bin_areas[7] = bin_areas[7] + 1.0
                        
            row_position = row_position + 1.0
         
        column_position = column_position + 1.0
        
    return bin_areas

        
        
        
        
        
'''
This function takes the edge image and crops around the area of interest 
(the largest contour point set (the object shape)), and then sends this image to 
be filled in
'''
def transform_to_binary_image(image):
    edged = image.copy()
    # find all contour point sets from an edge filter image
    image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    #find position in contour array of the largest area covered, to fetch this contour set
    position = find_biggest_contour(contours)
    
    #POTENTIAL ERROR NEEDS CATCHING HERE
    biggestContourSet = contours[position]

    contour_image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=1) 
    # crop filled contour image
    result = contour_image.copy()
    x,y,w,h = cv2.boundingRect(biggestContourSet)

    #result is the binary edge image cropped around the bounding box
    #of the biggest contour set 
    result = result[y:y+h, x:x+w]

    #fill image to be able to count shape area
    filled_image = fill_image(result)
    return filled_image
    
    
    
'''
This function is used to find the percentage area each bin takes up of the shape as a whole
'''    
def find_bin_percentages(input_image):
    #pre process the image, returns the edge image
    edged = pre_process(input_image, 50, 75)


    #closing the holes that may appear inside the edges to potentially close leaking
    #in the flood fill stage
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    #2x2 kernel  and one iteration to only fill potential gaps between exterior edges
    #so that the exterior contour points form a closed polygon and can be filled appropriately.
    kernel = np.ones((2,2),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations = 1)
  
    #obtain the filled cropped binary shape image
    filled_image = transform_to_binary_image(edged)


        
    ##find total pixel area of filled shape
    total_pixels, pixel_count = find_total_area(filled_image)
    print("Total area of white pixels in shape: " + str(pixel_count))
    completion_checker = (pixel_count * 100)/total_pixels
    print("Percentage of shape that is white" + str(completion_checker)+ "%") 
    
 
    
    #find centroid of the filled object shape
    cX, cY = get_object_centroid(filled_image)
    
    #calculate bin area for each bin
    bin_areas = find_bin_area(filled_image, cX, cY)
    
    #return each bin area count as a percentage area of the total shape area
    area_percentages = [0]*8
    count = 0
    for bin_area in bin_areas:
        area_percentages[count] = (bin_area * 100)/pixel_count
        count = count + 1

    return area_percentages, filled_image, pixel_count, cX, cY




'''
This function is to display meaningful output to the user to understand what the system is doing.
'''
def display(image, pixel_count, areas):
    count = 0
    for bin_area in areas:
        print("Area of shape inside bin " + str(count + 1) + ": " + str(bin_area) + "%")
        count = count + 1


    #draw each bin onto the image for clarity
    image_height, image_width = np.asarray(filled_image).shape
    #line coordinates
    start_point = (cX, 0) 
    end_point = (cX, image_height) 
    
    '''
    Each bin is plotted alongside the centroid onto the image so that the user can see
    where the bins have been split and how the central point of the captured image
    compared to the model may differ.
    '''
    display_image = np.asarray(filled_image).copy()
    #convert to rgb to show the red
    display_image = cv2.cvtColor(display_image,cv2.COLOR_GRAY2RGB)
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    start_point = (cX/2, 0) 
    end_point = (cX/2, image_height) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    start_point = (cX + cX/2, 0) 
    end_point = (cX + cX/2, image_height) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    start_point = (0, cY) 
    end_point = (image_width, cY) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 

    cv2.circle(display_image, (cX, cY), 3, (0, 0, 255), 0)
    #cv2.putText(filled_image, "Center", (cX - 25, cY - 5),cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)
   # cv2.imshow("Bin Segmentation mapped on cropped image", display_image)
               
    #cv2.waitKey(0)
    return display_image



'''
This function serves to calculate the discrepancy in proportional area between both
the model, and the captured image.
'''
def find_discrepancy(model_area_percentages, real_area_percentages):
    total_discrepancy = 0
    
    #potential to ignore total discrepancy if only one bin is hugely wrong?
    for x in range(len(model_area_percentages)):
        difference = abs(model_areas[x] - real_areas[x])
        total_discrepancy = total_discrepancy + difference
        
        
    '''
    
    Localised error recognition???
    
    
    '''
    
    print ('Total percentage discrepancy between two images is ' + str(total_discrepancy) + '%.')
    
    #currently 8 percent differnece in images allowed, allow 1 percent between each bin comparison
    if(total_discrepancy > 8):
        print ('Area discrepancy, potential misprint has occurred.')
    else:
        print ('Currently no area discrepancy between two shapes.')



#check for existence of input images
if os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
    expected_model_image = cv2.imread(sys.argv[1])
    captured_image = cv2.imread(sys.argv[2])
else:
    print ("Input path error, please review input arguments.")
    sys.exit()
    
image1 = cv2.imread('./testImages/empty.png')
image2 = cv2.imread('./testImages/benchy_failed_print.png')
image1 = cv2.resize(image1, (300,300))  
image2 = cv2.resize(image2, (300, 300))

image3 = image2 -image1
cv2.imshow("here", image3)
cv2.waitKey(0)

captured_image = image3

#images downsampled to reduce computational complexity on very large images. This can result in some lost detail
#this results in the same percentage difference between the shapes but drastically reduces computation time as
#there are less pixels to iterate over and count, this also reduces high frequency noise in the background
#of the image which could cause issues.
#input files should be the same resolution to reduce effect of this even further though
expected_model_image = cv2.resize(expected_model_image, (300,300))  
captured_image = cv2.resize(captured_image, (300, 300))

start_time = time.time()



  

#find the relative bin areas of botht the model image and the captured image
model_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(expected_model_image)
model_segmented = display(filled_image, pixel_count, model_areas)
real_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_image)
real_segmented = display(filled_image, pixel_count, real_areas)


#find the area discrepancy between the model and the captured image
find_discrepancy(model_areas, real_areas)

#this shows there is a marginal difference of around 1 second processing time without downsampling the input images
#with only a 0.3-0.5% accuracy sacrifice.
print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))


model_segmented = cv2.resize(model_segmented, (300,300))  
real_segmented= cv2.resize(real_segmented, (300, 300))  

horizontal = np.concatenate((model_segmented, real_segmented), axis = 1)
cv2.imshow("Comparison", horizontal)
cv2.waitKey(0)
print ('Complete.')

