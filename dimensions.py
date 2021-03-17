from cv2 import cv2
import numpy as np
import sys
import os.path
import time


'''
This function is used to obtain the edges of the shape.
'''
def pre_process(image, threshold_value_1, threshold_value_2):
    
    #convert the image to gray to reduce computational complexity as
    #only dealing with one colour channel
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    gray = np.asarray(gray)
    

    #blur the image, sufficiently enough to remove some of the higher frequency noise, and smooth the edges.
    gray = cv2.GaussianBlur(gray, (5,5), 0)
 
    ##changing thresholds doesn't affect area proportions but can affect  to fill shape and identify 
    ##edges with lower frequency change
    edges = cv2.Canny(gray, threshold_value_1, threshold_value_2)

    
    #closing the holes that may appear inside the edges to potentially close leaking
    #in the flood fill stage
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #3x3 kernel  and one iteration to only fill potential gaps between exterior edges
    #so that the exterior contour points form a closed polygon and can be filled appropriately. Keep this as minimal as possible to not
    #overly amplify shape differences
    kernel = np.ones((3, 3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    #returns the edge image
   
    return edges






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
This function serves to calculate the number of white pixels that exist
in each of the bins. It does this by splitting the shape into 8, and counting the 
white pixels in whichever bin it's currently inside. It returns each bin value in an
array
'''
def find_bin_area(image, cX, cY):
    
    image_height, image_width = image.shape
    image_height = image_height + 1
    image_width = image_width + 1
    values = np.asarray(image)
    bin_areas = [0] * 16
    row_position = 0
    column_position = 0
    
    for y in values:
        row_position = 0    
        for x in y:
            if(column_position < cY/2):
                if(x == 255):
                    if(row_position <= cX/2):
                        bin_areas[0] = bin_areas[0] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[1] = bin_areas[1] + 1.0
                    if(row_position >= cX and row_position <= (cX + (image_width + cX)/2)):
                        bin_areas[2] = bin_areas[2] + 1.0
                    if(row_position >= cX + (image_width + cX)/2):
                        bin_areas[3] = bin_areas[3] + 1.0
            if(column_position < cY and column_position > cY/2):
                    if(row_position <= cX/2):
                        bin_areas[4] = bin_areas[4] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[5] = bin_areas[5] + 1.0
                    if(row_position >= cX and row_position <= cX + (image_width + cX)/2):
                        bin_areas[6] = bin_areas[6] + 1.0
                    if(row_position >= (cX + (image_width + cX)/2)):
                        bin_areas[7] = bin_areas[7] + 1.0
            if(column_position > cY and column_position < cY + (image_height - cY/2)):
                if(x == 255):
                    if(row_position <= cX/2):
                        bin_areas[8] = bin_areas[8] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[9] = bin_areas[9] + 1.0
                    if(row_position >= cX and row_position <= cX + (image_width + cX)/2):
                        bin_areas[10] = bin_areas[10] + 1.0
                    if(row_position >= cX + (image_width + cX)/2):
                        bin_areas[11] = bin_areas[11] + 1.0
            if(column_position >  cY + (image_height - cY/2)):
                if(x == 255):
                    if(row_position <= cX/2):
                        bin_areas[12] = bin_areas[12] + 1.0
                    if(row_position <= cX and row_position >= cX/2):
                        bin_areas[13] = bin_areas[13] + 1.0
                    if(row_position >= cX and row_position <= cX + (image_width + cX)/2):
                        bin_areas[14] = bin_areas[14] + 1.0
                    if(row_position >= cX + (image_width + cX)/2):
                        bin_areas[15] = bin_areas[15] + 1.0
                        
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
    
    # find parent contour point set from an edge filter image
    contours= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    #can use 0 as using RETR_EXTERNAL which only fetches the largest contour set from the image
    biggestContourSet = contours[0]
    
    total_area = cv2.contourArea(biggestContourSet)
        
    contour_image = cv2.drawContours(edged, contours, -1, (255, 255, 255), thickness=1) 
    img = contour_image.copy()
    
    #closed polygon can be filled here
    img = cv2.fillPoly(img, pts = [biggestContourSet], color = 255)
    #fill image to be able to count shape area
    result = img.copy()
    x,y,w,h = cv2.boundingRect(biggestContourSet)

    #result is the binary edge image cropped around the bounding box
    #of the biggest contour set 
    result = result[y:y+h, x:x+w]
    
    filled_image = result
    return filled_image, total_area
    
    
    
'''
This function is used to find the percentage area each bin takes up of the shape as a whole
'''    
def find_bin_percentages(input_image):
    #pre process the image, returns the edge image
    edged = pre_process(input_image, 10, 90 )

    #obtain the filled cropped binary shape image
    filled_image, total_area = transform_to_binary_image(edged)

    print("Total area of white pixels in shape: " + str(total_area))
 
    
    #find centroid of the filled object shape
    cX, cY = get_object_centroid(filled_image)
    
    #calculate bin area for each bin
    bin_areas = find_bin_area(filled_image, cX, cY)
    
    #return each bin area count as a percentage area of the total shape area
    area_percentages = [0]*16
    count = 0
    
    for bin_area in bin_areas:
        area_percentages[count] = (bin_area * 100)/total_area
        count = count + 1

    return area_percentages, filled_image, total_area, cX, cY




'''
This function is to display meaningful output to the user to understand what the system is doing.
'''
def display(image, pixel_count, areas,  cX, cY):
 
    #draw each bin onto the image for clarity
   
    image_height, image_width = image.shape
    image_height = image_height + 1
    image_width = image_width + 1
    
    print (str(cX) + " " + str(cY))
    #line coordinates
    
    '''
    Each bin is plotted alongside the centroid onto the image so that the user can see
    where the bins have been split and how the central point of the captured image
    compared to the model may differ.
    '''
    start_point = (cX, 0) 
    end_point = (cX, image_height) 
    display_image = np.asarray(filled_image).copy()
    #convert to rgb to show the red
    display_image = cv2.cvtColor(display_image,cv2.COLOR_GRAY2RGB)
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    
    start_point = (int(cX/2), 0) 
    end_point = (int(cX/2), image_height) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    
    start_point = (cX + int((image_width - cX)/2), 0) 
    end_point = (cX + int((image_width - cX)/2), image_height)
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    
    start_point = (0, cY) 
    end_point = (image_width, cY) 
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
    
    start_point = (0, int(cY/2))
    end_point = (image_width, int(cY/2))
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 
     
    start_point = (0, int(cY +(image_height - cY)/2))
    end_point = (image_width, int(cY +(image_height - cY)/2))
    display_image = cv2.line(display_image, start_point, end_point, (0,0,255), 1) 

   

    cv2.circle(display_image, (cX, cY), 3, (0, 0, 255), 0)
    return display_image



'''
This function serves to calculate the discrepancy in proportional area between both
the model, and the captured image.
'''
def find_discrepancy(model_area_percentages, real_area_percentages):
    total_discrepancy = 0
    
    for x in range(len(model_area_percentages)):
        total_discrepancy += abs(model_area_percentages[x] - real_area_percentages[x])
        
    return total_discrepancy







#check for existence of input images
try:
    if os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
        expected_model_image = cv2.imread(sys.argv[1])
        captured_image = cv2.imread(sys.argv[2])
    
        #3rd arg is empty background
        if len(sys.argv) > 3 and os.path.isfile(sys.argv[2]) and os.path.isfile(sys.argv[3]):
            #optional background subtraction to remove background from the contour detection
            image1 = cv2.imread(sys.argv[2])
            image2 = cv2.imread(sys.argv[3])
            try:
                image3 = image2 - image1
                captured_image = image3
            except:
                print("Image dimensions do not match.")
                sys.exit()
    else:
        raise Exception
except:
    print ("Input path error, please review input arguments.")
    sys.exit()

    
    
#start runtime  
start_time = time.time()


#find the relative bin areas of botht the model image and the captured image
model_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(expected_model_image)
model_segmented = display(filled_image, pixel_count, model_areas, cX, cY)
real_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_image)
real_segmented = display(filled_image, pixel_count, real_areas,cX, cY)


#find the area discrepancy between the model and the captured image
difference = find_discrepancy(model_areas, real_areas)


#go down pyramid until potential match is found for images that may not have been detected properly
#doesn't matter that it amplifies errors for already false images, it's trying to find correct shapes.
if difference  > 7:
    print ("layer 1")
    captured_pyr_level_1 = cv2.pyrDown(captured_image)
    pyr_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_pyr_level_1)
    pyr_segmented = display(filled_image, pixel_count, pyr_areas, cX, cY)
    difference = find_discrepancy(model_areas, pyr_areas)
  
    if difference > 7:
        print ("layer 2")
        captured_pyr_level_2 = cv2.pyrDown(captured_pyr_level_1)
        pyr_areas, filled_image, pixel_count, cX, cY = find_bin_percentages(captured_pyr_level_2)
        pyr_segmented = display(filled_image, pixel_count, real_areas,cX, cY)
        difference = find_discrepancy(model_areas, pyr_areas)
        if(difference > 7):
            print ("layer 3")
            print ("Error detected. Area difference of: " + str(difference) + "%")
            print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))
            #cv2.namedWindow('Real', cv2.WINDOW_NORMAL)
            cv2.imshow("Real",pyr_segmented)
            #cv2.namedWindow('Expected', cv2.WINDOW_NORMAL)
            cv2.imshow("Expected",model_segmented)      
            cv2.waitKey(0)
        else:
            print ("No error detected. Area difference of: " + str(difference) + "%")
            print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))            
            cv2.namedWindow('Real', cv2.WINDOW_NORMAL)
            cv2.imshow("Real",pyr_segmented)
            #cv2.namedWindow('Expected', cv2.WINDOW_NORMAL)
            cv2.imshow("Expected",model_segmented)      
            cv2.waitKey(0)
        
    else:
        print ("No error detected. Area difference of: " + str(difference) + "%")
        print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))

         #cv2.namedWindow('Real', cv2.WINDOW_NORMAL)
        cv2.imshow("Real",pyr_segmented)
        #cv2.namedWindow('Expected', cv2.WINDOW_NORMAL)
        cv2.imshow("Expected",model_segmented)      
        cv2.waitKey(0)

else:
    print ("No error detected. Area difference of: " + str(difference) + "%")
    print ("--- Runtime: %s seconds. ---" % (time.time() - start_time))

     #cv2.namedWindow('Real', cv2.WINDOW_NORMAL)
    cv2.imshow("Real",real_segmented)
    #cv2.namedWindow('Expected', cv2.WINDOW_NORMAL)
    cv2.imshow("Expected",model_segmented)      
    cv2.waitKey(0)
    
