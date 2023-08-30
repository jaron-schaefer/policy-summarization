import sys
import os
import random
import csv
import pygame
import cv2
from PIL import Image
from pygame.locals import *
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pygaze import libtime, libscreen, libinput, eyetracker
from experiment_code.constants import *
from experiment_code.highscores.highscore_manager import *
from experiment_code.play_gridworld import PlayGridworld
from trials.trial import Trial
from experiment_code.textbox import TextInputBox
from experiment_code.experiment_utils import get_performance
from data.exp_data_utils import post_processing

total_reward = 0

def play_gridworld(grid_list, rewards, start_state, feature_imgs, summary = None, summary_grid_list = None):

    gridworld = PlayGridworld(grid_list, rewards, screen, disp)
    gridworld.reset(start_state, feature_imgs, summary, summary_grid_list)
    tracker.log(f"START_STATE: {start_state}")

    running = True
    while running:
        key, time = keyboard.get_key(flush=True)
        if key in ['q']:
            running = False
        elif key in ['escape']:
            end_experiment() 
            break
        elif key in ARROWKEYS:
            action = ARROWKEYS[key]
            next_state, reward, done = gridworld.step(action)
            next_state_identifier = gridworld.state_identifiers[next_state]
            tracker.log(f"ACTION_TAKEN: {action}")
            tracker.log(f"NEXT_STATE: {next_state_identifier}")

            if done:
                tracker.log("GOAL_STATE_REACHED")
                global total_reward
                total_reward += gridworld.total_reward
                libtime.pause(1000)
                running = False
    return gridworld.total_path, gridworld.total_reward
 

def run_trial(trial:Trial):

    # select feature images for this trial
    all_feature_imgs = sorted(os.listdir(LANDSCAPESDIR))
    all_feature_imgs.remove('0.png')
    #all_feature_imgs.remove('.DS_Store')
    random.shuffle(all_feature_imgs)
    randomized_feature_imgs = all_feature_imgs[:8]
    feature_imgs = ["0.png"] + randomized_feature_imgs

    tracker.log(f"START_TRIAL: {trial.trial_name}")
    tracker.log(f"FEATURE_IMAGES: {feature_imgs}")

    tracker.start_recording()
    tracker.log("START_GRIDWORLD_1")
    total_path_1, total_reward_1 = play_gridworld(trial.grid_list_1, trial.reward_weights, trial.start_state_gridworld_1, feature_imgs, trial.summary)
    tracker.log(f"TOTAL_PATH_TAKEN_GRIDWORLD_1: {total_path_1}")
    tracker.log(f"TOTAL_REWARD_GRIDWORLD_1: {total_reward_1}")
    tracker.log("END_GRIDWORLD_1")
    tracker.stop_recording()

    show_performance(total_reward_1, trial.optimal_reward_gridworld_1)
    
    tracker.start_recording()
    tracker.log("START_GRIDWORLD_2")
    total_path_2, total_reward_2 = play_gridworld(trial.grid_list_2, trial.reward_weights, trial.start_state_gridworld_2, feature_imgs, trial.summary, trial.grid_list_1)
    tracker.log(f"TOTAL_PATH_TAKEN_GRIDWORLD_2: {total_path_2}")
    tracker.log(f"TOTAL_REWARD_GRIDWORLD_2: {total_reward_2}")
    tracker.log("END_GRIDWORLD_2")
    tracker.stop_recording()

    show_performance(total_reward_2, trial.optimal_reward_gridworld_2)
    
    tracker.log(f"END_TRIAL: {trial.trial_name}")
    tracker.stop_recording()

    show_rocket_video()


def end_experiment():
    screen.clear()
    screen.draw_text(text="Transferring the data file, please wait...", fontsize=TEXTSIZE)
    disp.fill(screen)
    disp.show()
    # close connection to the tracker (this will close the data file, and copy it to the stimulus PC)
    tracker.close()
    # exit message
    screen.clear()
    screen.draw_text(text="This is the end of this experiment. Thank you for participating! \n\n (press any key to exit)", fontsize=TEXTSIZE)
    disp.fill(screen)
    disp.show()
    keyboard.get_key(keylist=None, timeout=None, flush=True)
    disp.close()
    libtime.expend()

def show_current_highscores(gamer_name, game_id):
    highscores = update_highscore(gamer_name,game_id,int(total_reward))
    longest_key_length = len(max(highscores.keys(), key=len))
    pos_from_mid = (longest_key_length + 5)*6

    screen.clear()
    screen.draw_text(text="Current Highscores", pos=(DISPSIZE[0]/2,DISPSIZE[1]/3),fontsize=40)
    ypos = 40
    for name in highscores.keys():
        if gamer_name in name:
            screen.draw_text(text=name, color = "green", pos=(DISPSIZE[0]/2-pos_from_mid, DISPSIZE[1]/3+ypos),center = False,  fontsize=TEXTSIZE)
            screen.draw_text(text=highscores[name], color = "green", pos=(DISPSIZE[0]/2+pos_from_mid, DISPSIZE[1]/3+ypos),center = False,  fontsize=TEXTSIZE)
        else:
            screen.draw_text(text=name, pos=(DISPSIZE[0]/2-pos_from_mid,DISPSIZE[1]/3+ypos),center = False, fontsize=TEXTSIZE)
            screen.draw_text(text=highscores[name], pos=(DISPSIZE[0]/2+pos_from_mid,DISPSIZE[1]/3+ypos),center = False,  fontsize=TEXTSIZE)
        ypos+= TEXTSIZE

    screen.draw_text(text = "\n\n (press any key to continue)",pos=(DISPSIZE[0]/2,DISPSIZE[1]/3+ypos), fontsize=TEXTSIZE)

    disp.fill(screen)
    disp.show()
    keyboard.get_key(keylist=None, timeout=None, flush=True)


def show_performance(reached_reward, optimal_reward):
    perc_of_optimal_path = get_performance(reached_reward, optimal_reward)

    screen.clear()
    if perc_of_optimal_path == 100:
        text = "Impressive work!!\n\nYou used the optimal path."
    elif perc_of_optimal_path > 75:
        text = "Congratulations!!\n\nYou took a highly efficient path."
    elif perc_of_optimal_path > 50:
        text = "Great job!!\n\nYou took a fairly efficient path."
    elif perc_of_optimal_path > 25:
        text = "Great effort!!\n\nYou took a somewhat efficient path."
    else:
        text = "Good effort!!\n\nYou're on your way to finding an efficient path."

    screen.draw_image(GASCANPATH, pos=(DISPSIZE[0]/2,DISPSIZE[1]/1.5), scale=4)
    screen.draw_text(text=str(perc_of_optimal_path)+ "%", color = "black", pos=(DISPSIZE[0]/2,DISPSIZE[1]/1.5), fontsize=50)
    screen.draw_text(text=text, pos=(DISPSIZE[0]/2,DISPSIZE[1]/3),fontsize=35)
    disp.fill(screen)
    disp.show()
    libtime.pause(3000)

def show_rocket_video():
    video_file = os.path.join(IMGDIR, "rocket_animation.mp4")
    cap = cv2.VideoCapture(video_file)
    success, img = cap.read()
    shape = img.shape[1::-1]
    clock = pygame.time.Clock()

    while success:
        clock.tick(60)
        success, img = cap.read()
        if not success:
            break
        frame_surface = pygame.image.frombuffer(img.tobytes(), shape, "BGR")
        frame_surface = pygame.transform.scale(frame_surface, (DISPSIZE[0], DISPSIZE[1]))
        screen.screen.blit(frame_surface, (0, 0))
        disp.fill(screen)
        disp.show()

def select_player_name():
    gamer_name_unique = False
    while not gamer_name_unique:
        gamer_name = TextInputBox(screen,
                                disp,
                                instruction= "\n Enter your player name",
                                supertext="Welcome to this experiment! \n If you want to participate in the highscore list, please choose a player name (not your real name). \n Otherwise just press Enter."
                                ).main_loop()
        gamer_name_unique = not check_if_name_exists(gamer_name)
    
    return gamer_name

#----------------------MAIN EXPERIMENT----------------------------------


# Enter a logfile name and create data directory
LOGFILENAME = ""
if not DUMMYMODE:
    while True:
        vp_code = input("Please enter the VP-Code: ")

        LOGFILEFOLDER = os.path.join(DATADIR, vp_code)
        if os.path.exists(LOGFILEFOLDER):
            print("\n---Code already exists!!!---\n")
            continue
    
        os.makedirs(LOGFILEFOLDER)

        LOGFILENAME = os.path.join(LOGFILEFOLDER, vp_code +".edf")
        break


libtime.expstart()
disp = libscreen.Display()
keyboard = libinput.Keyboard(keylist=["up","down","left","right","w","a","s","d","q","escape","space"], timeout=None)
screen = libscreen.Screen()


# --- SELECT GAMER NAME ---
gamer_name = select_player_name()
show_highscores = False
if gamer_name != "":
    show_highscores = True

# --- CALIBRATE EYETRACKER ---
screen.clear()
screen.draw_text(text=CALIBRATION_INSTRUCTIONS_E, fontsize=TEXTSIZE)
disp.fill(screen)
disp.show()

# initialize tracker and its logfile
tracker = eyetracker.EyeTracker(disp, data_file = LOGFILENAME)
keyboard.get_key(keylist=None, timeout=None, flush=True)

tracker.calibrate()

# --- RUN THE TEST TRIALS ---

screen.clear()
screen.draw_text(text="You will now play four test trials. \n Don't hesitate to ask any questions whenever you like.\n (press any key to continue)", fontsize=TEXTSIZE)
disp.fill(screen)
disp.show()
keyboard.get_key(keylist=None, timeout=None, flush=True)

tracker.log("START_TEST_TRIALS")

test_trials = ["4_LD_LF_HS_IRL.yaml","4_LD_HF_LS_IL.yaml","4_HD_LF_HS_IRL.yaml","4_HD_HF_HS_IL.yaml"]
for i,trial_name in enumerate(test_trials):
    tracker.drift_correction()
    trial=Trial()
    trial.get_trial(os.path.join(TESTTRIALDIR,trial_name))
    tracker.status_msg(str(i)+"/"+str(len(test_trials)))# send msg to tracker gui to display progress
    run_trial(trial)

tracker.log("END_TEST_TRIALS")

if show_highscores:
    show_current_highscores(gamer_name,"Test score")
total_reward = 0


# --- RUN THE MAIN TRIALS ---
screen.clear()
screen.draw_text(text=EXPERIMENT_START_TEXT, fontsize=TEXTSIZE)
disp.fill(screen)
disp.show()
keyboard.get_key(keylist=None, timeout=None, flush=True)
tracker.log("START_MAIN_TRIALS")

# get and shuffle trials
first_block = []
second_block = []

trials = os.listdir(TRIALDIR)

for filename in trials:
    if filename.startswith("0") and "IL" in filename:
        first_block.append(filename)
    elif filename.startswith("1") and "IRL" in filename:
        first_block.append(filename)
    elif filename.startswith("0") and "IRL" in filename:
        second_block.append(filename)
    elif filename.startswith("1") and "IL" in filename:
        second_block.append(filename)

random.shuffle(first_block)
random.shuffle(second_block)

trials = first_block + second_block

# run trials
for i,trial_name in enumerate(trials):
    tracker.status_msg(str(i)+"/"+str(len(trials)))
    tracker.drift_correction()
    tracker.status_msg(str(i)+"/"+str(len(trials)))# send msg to tracker gui to display progress
    trial=Trial()
    trial.get_trial(os.path.join(TRIALDIR,trial_name))
    run_trial(trial)
    # Display Gamescore if the current trial is a multiple of 4
    if (i + 1) % 4 == 0:  
        game_score_number = (i + 1) // 4  # Calculate the game score number based on the current trial 
        if show_highscores:
            show_current_highscores(gamer_name,"Game score " + str(game_score_number))
    # half of the experiment (calibrate again)
    if (i+1) == len(first_block):
        screen.clear()
        screen.draw_text(text="Great job! You have successfully completed the first half of the experiment.\n You can now take a well-deserved break. \nBefore you continue, we will calibrate the eye tracker to ensure accurate measurements for the second half of the study.\n\n (press any key to continue)", fontsize=TEXTSIZE)
        disp.fill(screen)
        disp.show()
        keyboard.get_key(keylist=None, timeout=None, flush=True)

        tracker.calibrate()


tracker.log("END_MAIN_TRIALS")


# --- END THE EXPERIMENT ---
end_experiment()

# run post processing
#if not DUMMYMODE:
#    post_processing(vp_code)