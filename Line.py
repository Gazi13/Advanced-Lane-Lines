# Define a class to receive the characteristics of each line detection
from collections import deque
import numpy as np

class Line():
    def __init__(self,bins):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.curvature = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # polynom
        self.poly = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # size of list
        self.bins = bins
        # poly list 
        self.poly_list = deque(maxlen=bins)
        # radiues list
        self.curv_list = deque(maxlen=bins)
        #passNumber
        self.passNumber = 1