from cv2 import cv2
import skimage
from PIL import Image, ImageDraw
import math
import imutils
from imutils import contours
import numpy as np
import sys
import os.path


#image pre processing
def pre_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    gray = np.asarray(gray)

    #blur the image, sufficiently enough to remove some of the higher frequency noise
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
 

    #cv2.imshow('blurred image', gray)
    ##changing thresholds doesn't affect area proportions but can affect affectives to fill shape and identify 
    ##edges with lower frequency change
    edges = cv2.Canny(gray, 50, 75)
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
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    
    
    row_position = 0
    column_position = 0
    for y in values:
        row_position = 0    
        for x in y:
            #print(x)
            if(column_position < cY):
                if(x == 255):
                    if(row_position <= cX):
                        bin1 = bin1 + 1.0
                    if(row_position >= cX):
                        bin2 = bin2 + 1.0
            if(column_position > cY):
                if(x == 255):
                    if(row_position <= cX):
                        bin3 = bin3 + 1.0
                    if(row_position >= cX):
                        bin4 = bin4 + 1.0
                        
            row_position = row_position + 1.0
         
        column_position = column_position + 1.0
        
    return bin1, bin2, bin3, bin4 
        
        
        
        
        
        
def transform_to_binary_image(image):
    edged = image.copy()
    # find all contour point sets from an edge filter image
    image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    #find position in contour array of the largest area covered, to fetch this contour set
    position = find_biggest_contour(contours)
    biggestContourSet = contours[position]

    contour_image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=1) 
    # crop filled contour image
    result = contour_image.copy()
    x,y,w,h = cv2.boundingRect(biggestContourSet)

    #result is the binary edge image cropped around the bounding box
    #of the biggest contour set
    result = result[y:y+h, x:x+w]

    #fill image
    filled_image = fill_image(result)
    return filled_image
    
    
    
        
def find_bin_percentages(input_image):
    #pre process the image
    processImage = pre_process(input_image)

    #get the edges of the image
    edged = processImage.copy()
  

    #closing the holes that may appear inside the edges to potentially speed up and avoid dilation in the next stage
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #2x2 kernel  and one iteration to only fill the shape until needed and not over fill
    kernel = np.ones((5,5),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations = 1)
    cv2.imshow('Edges' , edged)
    cv2.waitKey(0)

    #fill the image to try and create a binary model
    filled_image = transform_to_binary_image(edged)


    ## this dilates the edges image to fill any gaps in the outline so a binary 
    ## filled image of the object can be created, that's as close as possible to the model
    ## the model image will also have to be dilated the same amount of times to remove this discrepancy
    ## being an issue
    completion_checker = 0
    while(completion_checker < 20):
        
        ##find total pixel area of filled shape
        total_pixels, pixel_count = find_total_area(filled_image)
        print("Total area of white pixels in shape: " + str(pixel_count))
        completion_checker = (pixel_count * 100)/total_pixels
        print("Percentage of shape that is white" + str(completion_checker)+ "%") 
        
        #2x2 kernel  and one iteration to only fill the shape until needed and not over fill
        kernel = np.ones((2,2),np.uint8)
        edged = cv2.dilate(edged,kernel,iterations = 1)
        filled_image = transform_to_binary_image(edged)
        
        ##find total pixel area of filled shape after dilation
        total_pixels, pixel_count = find_total_area(filled_image)

            
    #cv2.imshow("FILLED", filled_image)
    cv2.imshow('Cropped Shape filled' , filled_image)
    cv2.waitKey(0)
    
    #find centroid of the filled object shape
    cX, cY = get_object_centroid(filled_image)
    
    
    bin1, bin2, bin3, bin4 = find_bin_area(filled_image, cX, cY)
    area_percentages = [(bin1 * 100)/pixel_count, (bin2 * 100)/pixel_count, (bin3 * 100)/pixel_count ,(bin4 * 100)/pixel_count]

    return area_percentages, filled_image, pixel_count, cX, cY





def display(image, pixel_count, areas):
    print("Area of shape inside bin 1: " + str(areas[0])+ "%")
    print("Area of shape inside bin 2: " + str(areas[1]) + "%")
    print("Area of shape inside bin 3: " + str(areas[2]) + "%")
    print("Area of shape inside bin 4: " + str(areas[3]) + "%")

    #draw each bin onto the image for clarity
    image_height, image_width = np.asarray(filled_image).shape
    #line coordinates
    start_point = (cX, 0) 
    end_point = (cX, image_height) 
    
    display_image = np.asarray(filled_image).copy()
    #convert to rgb to show the red
    display_image = cv2.cvtColor(display_image,cv2.COLOR_GRAY2RGB)
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 

    start_point = (0, cY) 
    end_point = (image_width, cY) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 

    cv2.circle(display_image, (cX, cY), 3, (0, 0, 255), 0)
    #cv2.putText(filled_image, "Center", (cX - 25, cY - 5),cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)
    print(display_image.shape)
    cv2.imshow("Bin Segmentation mapped on cropped image", display_image)
    cv2.waitKey(0)




def find_discrepancy(model_area_percentages, real_area_percentages):
    total_discrepancy = 0
    for x in range(4):
        difference = abs(model_areas[x] - real_areas[x])
        print difference
        total_discrepancy = total_discrepancy + difference
    
    print 'Total percentage discrepancy between two images is ' + str(total_discrepancy) + '%.'








#check for existence of input images
if os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
    expected_model_image = cv2.imread(sys.argv[1])
    captured_image = cv2.imread(sys.argv[2])
else:
    print ("Input path error.")
    sys.exit()



model_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(expected_model_image)
display(filled_image, pixel_count, model_areas)


real_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_image)
display(filled_image, pixel_count, real_areas)

find_discrepancy(model_areas, real_areas)



