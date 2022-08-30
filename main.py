import cv2 as cv
import numpy as np


# function to detect traffic lights using ML
def detect_lights_ML(img):
    # fetching the cascade files
    traffic_light_cascade = cv.CascadeClassifier("TrafficLight_HAAR_16Stages.xml")
    # detecting the traffic lights
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=2)
    # this aray contains the detected rectangles that are resized (a bit smaller because the detection makes them wider
    # than they really are)
    resized_lights = []
    # draw rectangles (to check if the detection worked fine)
    for rectangle in lights:
        x, y, w, h = rectangle.reshape(4)
        # the resize factor for the width is 0.6
        cv.rectangle(img, (int(x + 0.2 * w), y), (int(x + 0.2 * w) + int(w * 0.6), y + h), (0, 0, 255), 2)
        resized_lights.append(np.array([int(x + 0.2 * w), y, int(w * 0.6), h]))
    return resized_lights, img


# function to detect the brightest circle in the given areas
def detect_each_traffic_light_color(lights, img):
    # list that will have the color of each traffic light
    colors = []
    # iterate over all the lights found
    for light in lights:
        x, y, w, h = light.reshape(4)
        # mask the region of the traffic light
        mask = np.zeros(img.shape[:2], dtype='uint8')
        # put the pixels white only if they are in the plate area
        for i in range(x, x + w):
            for j in range(y, y + h):
                mask[j][i] = 255
        masked = cv.bitwise_and(img, img, mask=mask)

        # transform image to ease circle detection
        blured = cv.GaussianBlur(masked, (5, 5), 0)
        gray = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)

        # show masked and transformed image
        # cv.imshow("result",thresh)
        # cv.waitKey(0)

        # detect the color and append it to the list of colors
        detected_color = check_traffic_light_color(img, thresh, 0.2, w)
        colors.append(detected_color)
    return colors


# function to detect the color of a single traffic light
def check_traffic_light_color(img, thresh, percentage, width):
    # use different values of param2 until a circle with adapted color matches
    for param2 in range(40, 0, -2):
        circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT,
                                  1, 45, param1=75, param2=param2, minRadius=int(width / 20), maxRadius=int(width / 2));
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the found circles
            for circle in circles:
                # check the color of the circle
                color = color_domination(percentage, circle, img)
                if color != "Unknown":
                    return color
    # if no color found, return unknown
    return "Unkown"


# function to check if the given pixel is in the given color range
def in_color_range(pixel, color):
    return (color[0][0] <= pixel[0] <= color[0][1]) and (color[1][0] <= pixel[1] <= color[1][1]) and (
            color[2][0] <= pixel[2] <= color[2][1])


# function to check the color of a circle
def color_domination(percentage, circle, img):
    # the 3 first intervals of each color correspond to the traffic lights possible BRG colors range the last value
    # in each list is the number of pixels within the circle that are of this color(initiated at 0 at first)
    orange = [[0, 100], [100, 200], [200, 250], 0, "orange"]
    green = [[100, 255], [193, 255], [0, 240], 0, "green"]
    red = [[0, 100], [0, 100], [180, 255], 0, "red"]
    colors = [orange, green, red]
    # the total number of pixels in the circle
    total_cnt = 0

    # the idea is first to go through each pixel in the square in which the circle is inscribed
    (x, y, r) = circle
    min_x = min(0, x - r)
    max_x = max(len(img[0]), x + r)
    min_y = min(0, y - r)
    max_y = max(len(img), y + r)

    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            # check if the pixel is actually inside the disque
            if ((i - x) ** 2 + (j - y) ** 2) < r ** 2:
                total_cnt += 1
                # check if the pixel is in the color range of one of the three possible colors
                for color in colors:
                    if in_color_range(img[j][i], color):
                        color[3] += 1

    # check if in the circle a color is dominant and more present than the given percentage in parameter
    # value that will store the maximum color percentage
    maximum = 0
    # value that will store the dominant color
    result = "Unknown"
    for color in colors:
        if (color[3] / total_cnt) > maximum:
            maximum = (color[3] / total_cnt)
            result = color[4]
    if maximum > percentage:
        return result

    # if there is no dominant color, we return "Unknown"
    return "Unknown"


# function to display the color result for each traffic light detected
def display_results(img, lights, colors):
    # iterate through the colors found
    color_it = iter(colors)
    # iterate through the traffic lights
    for light in lights:
        x, y, w, h = light.reshape(4)
        cv.putText(img, next(color_it), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 2)
    return img


# test with image
img = cv.imread("multi_lights.jpg")
# resize it because too big
resized = cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation=cv.INTER_AREA)
# detect traffic lights with ML
lights, result = detect_lights_ML(img)
# detect the color of each traffic light
colors = detect_each_traffic_light_color(lights, img)
# display the results
result = display_results(result, lights, colors)
# save the result
cv.imwrite("result.jpg", result)
# display the results image
cv.imshow("result", result)
cv.waitKey(0)
