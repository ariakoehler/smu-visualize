import os, sys, inspect
import numpy as np


sys.path.insert(0, os.path.join(os.getcwd(), '../lib/'))

import Leap


class SampleListener(Leap.Listener):

    def on_connect(self, controller):
        print('Connected')

    def on_frame(self, controller):
        print('Frame available')
        


def main():
    listener = SampleListener()
    controller = Leap.Controller()

    controller.add_listener(listener)
    
    print('Press enter to quit.')
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)


if __name__=='__main__':
    main()
