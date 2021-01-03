from cv2 import cv2
import skimage
from PIL import Image, ImageDraw
import math
import imutils
from imutils import contours
import numpy as np


#image pre processing
def pre_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    gray = np.asarray(gray)

    #blur the image
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #cv2.imshow('blurred image', gray)
    edges = cv2.Canny(gray, 20, 100)
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
    x,y,w,h = cv2.boundingRect(filled_image)
    cv2.drawContours(filled_image, [contours[0]], 0, 255, -1)
    return filled_image

#find the total area taken up by the object
def find_total_area(image):
    pixel_values = np.asarray(filled_image)
    pixel_count = 0
    total_pixels = 0

    for pixel_row in pixel_values:
        for pixel_value in pixel_row:
            #check if the pixel is white
            if(pixel_value == 255):
                pixel_count = pixel_count + 1
            total_pixels = total_pixels + 1
    
    return total_pixels, pixel_count
        
        
        
        
    
    
    
    
#image = cv2.imread("bank.jpg")

expected_model_image = cv2.imread("circle_with_mess.jpg")

#pre process the image
processImage = pre_process(expected_model_image)

#cv2.imshow('edges ', edges)
edged = processImage.copy()


# find all contour point sets from an edge filter image
image2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 


#find position in contour array of the largest area covered, to fetch this contour set
position = find_biggest_contour(contours)
biggestContourSet = contours[position]


#get central point from the biggest set of contour points
cX, cY = get_object_centroid(biggestContourSet)



#draw bounding box
contourBoundingBox = cv2.minAreaRect(biggestContourSet)
box = cv2.boxPoints(contourBoundingBox)
#print(box)
box = np.intc(box)


cv2.imshow('Shape bounding box' ,cv2.drawContours(expected_model_image,[box],0,(0,0,255),2))
cv2.waitKey(0)

centroidImage = edged.copy()
cv2.circle(centroidImage, (cX, cY), 1, (255, 255, 255), 0)
cv2.putText(centroidImage, "Center", (cX - 25, cY - 5),cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
cv2.imshow("Centroid mapped", centroidImage)
cv2.waitKey(0)


im3 = cv2.drawContours(image2, contours, -1, (255, 255, 255), thickness=1) 
# crop filled contour image
result = im3.copy()
x,y,w,h = cv2.boundingRect(biggestContourSet)

#result is the binary edge image cropped around the bounding box
#of the biggest contour set
result = result[y:y+h, x:x+w]


#fill image
filled_image = fill_image(result)



#cv2.imshow("FILLED", filled_image)
cv2.imshow('Cropped Shape filled' , filled_image)
cv2.waitKey(0)


##find total pixel area of filled shape
total_pixels, pixel_count = find_total_area(filled_image)
    
print("Total area of white pixels in shape: " + str(pixel_count))
print("Area of one quarter: " + str((pixel_count * 100/4)/pixel_count) + "%")

#find centroid of the filled object shape
cX, cY = get_object_centroid(filled_image)






#find area in each bin
image_height, image_width = np.asarray(filled_image).shape
#line coordinates
start_point = (cX, 0) 
end_point = (cX, image_height) 
  
display_image = np.asarray(filled_image).copy()
display_image = cv2.cvtColor(display_image,cv2.COLOR_GRAY2RGB)
display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 

start_point = (0, cY) 
end_point = (image_width, cY) 
display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 



cv2.circle(display_image, (cX, cY), 1, (0, 0, 255), 0)
#cv2.putText(filled_image, "Center", (cX - 25, cY - 5),cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)
cv2.imshow("Bin Segmentation mapped on cropped image", display_image)
cv2.waitKey(0)
