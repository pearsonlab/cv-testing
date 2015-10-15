'''
Display test for PyschoPy.  We will take a video of the screen as it 
displays this test, then we will attempt to extract the screen from
the video.
'''

from psychopy import visual, core


def main():
    testWin = visual.Window(
            size=(1280, 800), monitor="tobiiMonitor", units="pix", screen=0,
            fullscr=True)
    x = -0.5
    y = -0.5
    dotStim = visual.Circle(testWin, size=(0.1, 0.1*(8.0/5)), fillColor=[0, 1, 0],
                            units='norm', pos=(x, y), autoDraw=True)
    testWin.flip()
    core.wait(4)
    for i in range(100):
        x = x+0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        y = y+0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        x = x-0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        y = y-0.01
        dotStim.pos = (x, y)
        testWin.flip()
    testWin.close()

if __name__ == '__main__':
    main()
