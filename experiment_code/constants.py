import platform
import os
import sys
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_code.experiment_utils import convert_visual_degree_to_pixels

# system = platform.system()

# if system == "Darwin":
#     from AppKit import NSScreen
#     main_screen = NSScreen.mainScreen()
#     screen_width = int(main_screen.frame().size.width)
#     screen_height = int(main_screen.frame().size.height)
# else:
screen_width = 2560
screen_height = 1440



### --- PATHS AND DIRECTORIES --- ###

# full path to the current directory
DIR = os.path.dirname(__file__)

# Path to the directory where data files will be stored
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'experiment_data')

# Path to the directory that contains the image files for drawing the gridworld
IMGDIR = os.path.join(DIR, 'images')

# Path to the directory where the landscapes are save
LANDSCAPESDIR = os.path.join(IMGDIR, 'landscapes')

# Path to the directory where there arrwo images are saved (used for summary)
ARROWSDIR = os.path.join(IMGDIR, 'arrows')
STARTARROWSDIR = os.path.join(IMGDIR, 'start_arrows')

# Path to the agent image
AGENTIMGPATH = os.path.join(IMGDIR, 'agent.png')

# Path to the agent image
GASCANPATH = os.path.join(IMGDIR, 'gas_can.png')

# Path to the directory where all trials are saved
TRIALDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trials', 'experiment_trials')
TESTTRIALDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trials', 'test_trials')

# Path to the file where the demographic information will be stored
DEMOGRAPHIC_INFORMATION_FILE = os.path.join(DATADIR, 'demographical_infos.csv')



### --- EYETRACKER --- ###
DUMMYMODE = True # False for gaze contingent display, True for dummy mode (using mouse or joystick)
TRACKERTYPE = 'eyelink' # either 'smi', 'eyelink' or 'dummy' (NB: if DUMMYMODE is True, trackertype will be set to dummy automatically)
#EVENTDETECTION = 'pygaze'
EYELINKCALBEEP = False


DISPTYPE = 'pygame' # either 'psychopy' or 'pygame'
DISPSIZE = (screen_width,screen_height) # canvas size
SCREENSIZE = (59.5, 33.5) #nthe physical screen size in mm. LAB: (595, 335).
SCREENDIST = 110.0 # the distance in mm between the participant and the display. LAB: 1000
MOUSEVISIBLE = False # mouse visibility
FULLSCREEN = True
TEXTSIZE = 30


# Colors 
BGC = (125,125,125,255) # backgroundcolour
FGC = (0,0,0,255) # foregroundcolour


ARROWKEYS = {
    'up': 0,     # UP
    'down': 1,   # DOWN
    'left': 2,   # LEFT
    'right': 3,  # RIGHT
    'w': 0,      # UP
    's': 1,      # DOWN
    'a': 2,      # LEFT
    'd': 3       # RIGHT
}



### --- Gridworld coordinates ---###

max_x = 7
max_y = 5
summary_length = 5

SQUARE_SIZE = 150 # round(convert_visual_degree_to_pixels(2, screen_width, SCREENSIZE[0]*100, SCREENDIST*100)) 
SUMMARY_SQUARE_SIZE= 60 #int(SQUARE_SIZE /2.5)
SUMMARY_GAP_SIZE = SUMMARY_SQUARE_SIZE

GRID_WIDTH = max_x * SQUARE_SIZE
GRID_HEIGHT = max_y * SQUARE_SIZE

#Position from where to draw the gridworld surface
GRID_X = (screen_width - GRID_WIDTH) // 2
GRID_Y = (screen_height - GRID_HEIGHT) // 2 + SQUARE_SIZE

LENGTH_SUMMARY_SURFACE = (max_x * SUMMARY_SQUARE_SIZE * summary_length) + ((summary_length-1)*SUMMARY_GAP_SIZE) + 2 * SUMMARY_SQUARE_SIZE
HEIGHT_SUMMARY_SURFACE = SUMMARY_SQUARE_SIZE * max_y + 2 * SUMMARY_SQUARE_SIZE 

SUMMARY_X = screen_width/2 - LENGTH_SUMMARY_SURFACE/2
SUMMARY_Y = (GRID_Y/2) - (HEIGHT_SUMMARY_SURFACE/2)

#Position from where to draw the summary surface
SUMMARY_POS = (SUMMARY_X, SUMMARY_Y)



### --- Instruction texts --- ###

CALIBRATION_INSTRUCTIONS_E = """
We will now calibrate the EyeTracker.
If you wear glasses, make sure to remove them now and use your contact lenses instead.
First, you will need to fixate a stimulus (black circle) in the center of the screen.
Once you are ready, we will start the calibration. Then, multiple calibration stimuli 
will appear in the corners of the screen. Your task is to fixate on them as quickly as possible.\n
If you have any questions beforehand, let us know.
\n
Press any key to continue...
"""

EXPERIMENT_START_TEXT = """
The test trials have ended.\n
If you still have any questions to ask, feel free to ask now.\n
Otherwise, the main experiment can begin.\n
\n
Press any key to start the experiment...
"""