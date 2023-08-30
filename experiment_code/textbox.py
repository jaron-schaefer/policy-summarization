import pygame
from pygame.locals import *
import sys
from typing import List
from constants import TEXTSIZE, BGC, DISPSIZE


class TextInputBox(object):
    """
        Class to draw a text input box.
    """

    def __init__(self, 
                 screen, 
                 display,
                 instruction,
                 subtext: str = "Press Enter to continue...",
                 supertext: str = None,
                 key_list: List[str] = None,
                 caps: bool = False) -> None:
        """
            Constructor for the text box.

            :param screen: The screen to draw upon (Pygame Surface)
            :param font: The font to use for rendering text
            :param instruction: The instructions to display before the text input
            :param subtext: Optional subtext to display under the textbox
            :param supertext: Optional supertext to display over the textbox
            :param key_list: List of string characters that are allowed for use
            :param caps: Whether all letters should be capitalized
        """

        # Initialize attributes
        self.pygame_surface = screen.screen
        self.screen = screen
        self.display = display
        self.font = pygame.font.SysFont(None, TEXTSIZE) 
        self.text = ''
        self.instruction = instruction
        self.caps = caps
        self.subtext = subtext
        self.supertext = supertext

        # Default key List: low-case, upper-case, and numbers
        if key_list is None:
            self.key_list = [chr(x) for x in range(65, 90 + 1)]
            self.key_list.extend([chr(y) for y in range(97, 122 + 1)])
            self.key_list.extend([chr(z) for z in range(48, 57 + 1)])
        else:
            self.key_list = key_list

    def main_loop(self):
        """
            Runs the main loop for the Textbox. This terminates once Enter is pressed.
        """

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # Clear the pygame_surface
            self.pygame_surface.fill(BGC)
            
            # Calculate the center coordinates of the screen and text surfaces
            screen_center_x = DISPSIZE[0] // 2
            screen_center_y = DISPSIZE[1] // 2
   
            # Render text surfaces on the screen
            self.screen.draw_text(f'{self.instruction}: {self.text}',pos = (screen_center_x, screen_center_y), fontsize= TEXTSIZE)
            self.screen.draw_text(self.subtext,pos = (screen_center_x, screen_center_y + 230), fontsize= TEXTSIZE)
            if self.supertext is not None:
                self.screen.draw_text(self.supertext,pos = (screen_center_x, screen_center_y - 190), fontsize= TEXTSIZE)


            self.display.fill(self.screen)
            self.display.show()

            # If any of the keys is pressed, append it to the answer
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.unicode in self.key_list:
                        # Upper-case if any shift is pressed
                        if event.mod & KMOD_SHIFT or self.caps:
                            letter = event.unicode.upper()
                        else:
                            letter = event.unicode
                        # Append to answer
                        self.text += letter

                    # Add whitespace if space is pressed
                    elif event.key == K_SPACE:
                        self.text += " "

                    # Delete the last entry of the answer if the backspace is pressed
                    elif event.key == K_BACKSPACE:
                        self.text = self.text[:-1]

                    # Submit the answer if the Enter key is pressed
                    elif event.key == K_RETURN:
                        return self.text
        


# # Pygame initialization


# disp = libscreen.Display()
# screen = libscreen.Screen()


# # Create an instance of the TextInputBox
# text_box = TextInputBox(screen, disp, "Enter your name:",supertext="Halloooooo")

# # Main loop
# name = text_box.main_loop()
# print("Name entered:", name)
