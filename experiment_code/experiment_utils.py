import math

def convert_visual_degree_to_pixels(visual_degree, display_width_pixels, screen_width_mm, viewing_distance_mm):
    """Converts visual degree to target position in pixels.

    :param visual_degree: The visual degree to convert.
    :type visual_degree: float
    :param display_width_pixels: The width of the display in pixels.
    :type display_width_pixels: int
    :param screen_width_mm: The width of the screen in mm.
    :type screen_width_mm: float
    :param viewing_distance_mm: The viewing distance in mm.
    :type viewing_distance_mm: float
    :return: The target position in pixels.
    :rtype: float
    """
    # Calculate pixels per mm
    pixels_per_mm = display_width_pixels / screen_width_mm
    
    # Calculate the target position in mm
    target_position_mm = math.tan(math.radians(visual_degree)) * viewing_distance_mm
    
    # Calculate the target position in pixels
    target_position_pixels = target_position_mm * pixels_per_mm
    
    return target_position_pixels


def compute_visual_degree_for_pixels(pixels, display_width_pixels, screen_width_mm, viewing_distance_mm):
    """Converts target position in pixels to visual degree.

    :param pixels: The target position in pixels.
    :type pixels: float
    :param display_width_pixels: The width of the display in pixels.
    :type display_width_pixels: int
    :param screen_width_mm: The width of the screen in mm.
    :type screen_width_mm: float
    :param viewing_distance_mm: The viewing distance in mm.
    :type viewing_distance_mm: float
    :return: The visual degree.
    :rtype: float
    """
    # Calculate pixels per mm
    pixels_per_mm = display_width_pixels / screen_width_mm
    
    # Calculate the target position in mm
    target_position_mm = pixels / pixels_per_mm
    
    # Calculate the visual degree
    visual_degree = math.degrees(math.atan(target_position_mm / viewing_distance_mm))
    
    return visual_degree


def get_sim_to_summary(trajectory, summary):
    # Get all states used in summary
    all_summary_states = set()
    for summary_trajectory in summary:
        for state_indice, action in summary_trajectory:
            all_summary_states.add(state_indice)

    trajectory_states = [state_indice for state_indice, action in trajectory]

    common_elements = len(set(trajectory_states) & all_summary_states)

    # Calculate the percentage of same states
    percentage = common_elements / len(trajectory_states)

    return percentage

def get_sim_between_trajectories(trajectory_1, trajectory_2):
    trajectory_1_states = set([state_indice for state_indice, action in trajectory_1])
    trajectory_2_states = set([state_indice for state_indice, action in trajectory_2])

    common_elements = len(trajectory_1_states.intersection(trajectory_2_states))
    total_states = len(trajectory_1_states.union(trajectory_2_states))

    # Calculate the percentage of shared states
    percentage = (common_elements / total_states)

    return percentage

def get_performance(reached_reward, optimal_reward):
    perc_of_optimal_path = round((reached_reward / optimal_reward) * 100)
    if perc_of_optimal_path < 0:
        perc_of_optimal_path = 0
    
    return perc_of_optimal_path

