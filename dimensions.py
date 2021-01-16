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


#image pre processing
def pre_process(image, threshold_value_1, threshold_value_2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    gray = np.asarray(gray)

    #blur the image, sufficiently enough to remove some of the higher frequency noise
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
 

    #cv2.imshow('blurred image', gray)
    ##changing thresholds doesn't affect area proportions but can affect affectives to fill shape and identify 
    ##edges with lower frequency change
    edges = cv2.Canny(gray, threshold_value_1, threshold_value_2)
    return edges






#find the contour from the set with the largest area
def find_biggest_contour(contours):
    highest = 0
    position = 0
    i = 0
    
    for c in contours:
        area = cv2.contourArea(c)
       # print(area)
        if(area > highest):
            highest = area
            position = i
        i += 1
    return position






#find the central point from the set of contour points
def get_object_centroid(image):
    #find the moments of the binary image
    imageMoments = cv2.moments(image)
    x = int(imageMoments["m10"] / imageMoments["m00"])
    y =  int(imageMoments["m01"] / imageMoments["m00"])
    
    #return co ordinates of central point
    return x, y
    
    
    
    


#flood the inside of the exterior object edge with white
def fill_image(image):
    filled_image = np.zeros_like(image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(filled_image, [contours[0]], 0, 255, -1)
    return filled_image






#find the total area taken up by the object
def find_total_area(image):
    pixel_values = np.asarray(image)
    pixel_count = 0
    total_pixels = 0

    for pixel_row in pixel_values:
        for pixel_value in pixel_row:
            #check if the pixel is white
            if(pixel_value == 255):
                pixel_count = pixel_count + 1
            total_pixels = total_pixels + 1
    
    return total_pixels, pixel_count






##only currently first bin
def find_bin_area(image, cX, cY):
    values = np.asarray(image)
    bin_areas = [0] * 8
    
    row_position = 0
    column_position = 0
    for y in values:
        row_position = 0    
        for x in y:
            #print(x)
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
    
    
    
        
def find_bin_percentages(input_image):
    #pre process the image
    processImage = pre_process(input_image, 50, 75)

    #get the edges of the image
    edged = processImage.copy()
  

    #closing the holes that may appear inside the edges to potentially speed up and avoid dilation in the next stage
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #2x2 kernel  and one iteration to only fill the shape until needed and not over fill
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations = 1)
    #cv2.imshow('Edges' , edged)
    #cv2.waitKey(0)

    #fill the image to try and create a binary model
    filled_image = transform_to_binary_image(edged)


    ## this dilates the edges image to fill any gaps in the outline so a binary 
    ## filled image of the object can be created, that's as close as possible to the model
    ## the model image will also have to be dilated the same amount of times to remove this discrepancy
    ## being an issue
  
        
    ##find total pixel area of filled shape
    total_pixels, pixel_count = find_total_area(filled_image)
    print("Total area of white pixels in shape: " + str(pixel_count))
    completion_checker = (pixel_count * 100)/total_pixels
    print("Percentage of shape that is white" + str(completion_checker)+ "%") 
    
    #2x2 kernel  and one iteration to only fill the shape until needed and not over fill, further ensures the edges form a closed polygon
    kernel = np.ones((2,2),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations = 1)
    filled_image = transform_to_binary_image(edged)
    
    ##find total pixel area of filled shape after dilation
    total_pixels, pixel_count = find_total_area(filled_image)


    #cv2.imshow('Cropped Shape filled' , filled_image)
    #cv2.waitKey(0)
    
    #find centroid of the filled object shape
    cX, cY = get_object_centroid(filled_image)
    
    bin_areas = find_bin_area(filled_image, cX, cY)
    
    #return each bin area count as a percentage area of the total shape area
    area_percentages = [0]*8
    count = 0
    for bin_area in bin_areas:
        area_percentages[count] = (bin_area * 100)/pixel_count
        count = count + 1

    return area_percentages, filled_image, pixel_count, cX, cY





def display(image, pixel_count, areas):
    count = 0
    print image.shape
    for bin_area in areas:
        print("Area of shape inside bin " + str(count + 1) + ": " + str(bin_area) + "%")
        count = count + 1


    #draw each bin onto the image for clarity
    image_height, image_width = np.asarray(filled_image).shape
    #line coordinates
    start_point = (cX, 0) 
    end_point = (cX, image_height) 
    
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




def find_discrepancy(model_area_percentages, real_area_percentages):
    total_discrepancy = 0
    
    #potential to ignore total discrepancy if only one bin is hugely wrong?
    for x in range(len(model_area_percentages)):
        difference = abs(model_areas[x] - real_areas[x])
        total_discrepancy = total_discrepancy + difference
    
    print 'Total percentage discrepancy between two images is ' + str(total_discrepancy) + '%.'
    
    #currently 8 percent differnece in images allowed, allow 1 percent between each bin comparison
    if(total_discrepancy > 8):
        print 'Area discrepancy, potential misprint has occurred.'
    else:
        print 'Currently no area discrepancy between two shapes.'







#check for existence of input images
if os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
    expected_model_image = cv2.imread(sys.argv[1])
    captured_image = cv2.imread(sys.argv[2])
else:
    print ("Input path error, please review input arguments.")
    sys.exit()

#images downsampled to reduce computational complexity on very large images. This can result in some lost detail
#this results in the same percentage difference between the shapes but drastically reduces computation time as
#there are less pixels to iterate over and count, this also reduces high frequency noise in the background
#of the image which could cause issues.
#input files should be the same resolution to reduce effect of this even further though
expected_model_image = cv2.resize(expected_model_image, (300,300))  
captured_image = cv2.resize(captured_image, (300, 300))

horizontal = np.concatenate((expected_model_image, captured_image), axis = 1)
cv2.imshow("Initial Images", horizontal)
cv2.waitKey(0)
start_time = time.time()



  

#find the relative bin areas of botht the model image and the captured image
model_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(expected_model_image)
model_segmented = display(filled_image, pixel_count, model_areas)
real_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_image)
real_segmented = display(filled_image, pixel_count, real_areas)

find_discrepancy(model_areas, real_areas)

#this shows there is a marginal difference of around 1 second processing time without downsampling the input images
#with only a 0.3-0.5% accuracy sacrifice.
print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))

model_segmented = cv2.resize(model_segmented, (300,300))  
real_segmented= cv2.resize(real_segmented, (300, 300))  

horizontal = np.concatenate((model_segmented, real_segmented), axis = 1)
cv2.imshow("Comparison", horizontal)
cv2.waitKey(0)
print model_segmented.shape
print real_segmented.shape


