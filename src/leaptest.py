import os, sys, inspect
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd(), '../lib/'))

import Leap


class SampleListener(Leap.Listener):

    recent_hand = None
    recent_frame = None

    def on_connect(self, controller):
        print('Connected')

    def on_frame(self, controller):
        # Check that frame was retrieved
        #print('Frame available.')

        # get new frame and log it in member data
        frame = controller.frame()
        self.recent_frame = frame

        # get hands in frame, update most recent hand if only one
        handlist = frame.hands
        if len(handlist) is 1 and handlist[0].id is not self.recent_hand:
            self.recent_hand = handlist[0].id
            
    def get_hand_data(self):
        if self.recent_frame is not None and self.recent_hand is not None:
            return self.recent_frame.hand(self.recent_hand).stabilized_palm_position.to_tuple(), self.recent_frame.hand(self.recent_hand).palm_normal.to_tuple(), self.recent_frame.hand(self.recent_hand).direction.to_tuple()
        else:
            return (0,0,0), (0,0,0), (0,0,0)


def main():
    listener = SampleListener()
    controller = Leap.Controller()

    controller.add_listener(listener)

    while True:
        print(listener.get_hand_data())
    
    print('Press enter to quit.')
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)


if __name__=='__main__':
    main()
