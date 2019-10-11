from __future__ import print_function, division
from math import sin, cos, radians
import numpy as np
import matplotlib as plt
import WalabotAPI
try:  # for Python 2
    import Tkinter as tk
except ImportError:  # for Python 3
    import tkinter as tk
try:  # for Python 2
    range = xrange
except NameError:
    pass


APP_X, APP_Y = 50, 50  # location of top-left corner of app window
CANVAS_LENGTH = 350  # in pixels
COLORS = ["blue", "green", "red", "yellow", "purple"]  # of different targets
COLORS2 = [
    "000083", "000087", "00008B", "00008F", "000093", "000097", "00009B",
    "00009F", "0000A3", "0000A7", "0000AB", "0000AF", "0000B3", "0000B7",
    "0000BB", "0000BF", "0000C3", "0000C7", "0000CB", "0000CF", "0000D3",
    "0000D7", "0000DB", "0000DF", "0000E3", "0000E7", "0000EB", "0000EF",
    "0000F3", "0000F7", "0000FB", "0000FF", "0003FF", "0007FF", "000BFF",
    "000FFF", "0013FF", "0017FF", "001BFF", "001FFF", "0023FF", "0027FF",
    "002BFF", "002FFF", "0033FF", "0037FF", "003BFF", "003FFF", "0043FF",
    "0047FF", "004BFF", "004FFF", "0053FF", "0057FF", "005BFF", "005FFF",
    "0063FF", "0067FF", "006BFF", "006FFF", "0073FF", "0077FF", "007BFF",
    "007FFF", "0083FF", "0087FF", "008BFF", "008FFF", "0093FF", "0097FF",
    "009BFF", "009FFF", "00A3FF", "00A7FF", "00ABFF", "00AFFF", "00B3FF",
    "00B7FF", "00BBFF", "00BFFF", "00C3FF", "00C7FF", "00CBFF", "00CFFF",
    "00D3FF", "00D7FF", "00DBFF", "00DFFF", "00E3FF", "00E7FF", "00EBFF",
    "00EFFF", "00F3FF", "00F7FF", "00FBFF", "00FFFF", "03FFFB", "07FFF7",
    "0BFFF3", "0FFFEF", "13FFEB", "17FFE7", "1BFFE3", "1FFFDF", "23FFDB",
    "27FFD7", "2BFFD3", "2FFFCF", "33FFCB", "37FFC7", "3BFFC3", "3FFFBF",
    "43FFBB", "47FFB7", "4BFFB3", "4FFFAF", "53FFAB", "57FFA7", "5BFFA3",
    "5FFF9F", "63FF9B", "67FF97", "6BFF93", "6FFF8F", "73FF8B", "77FF87",
    "7BFF83", "7FFF7F", "83FF7B", "87FF77", "8BFF73", "8FFF6F", "93FF6B",
    "97FF67", "9BFF63", "9FFF5F", "A3FF5B", "A7FF57", "ABFF53", "AFFF4F",
    "B3FF4B", "B7FF47", "BBFF43", "BFFF3F", "C3FF3B", "C7FF37", "CBFF33",
    "CFFF2F", "D3FF2B", "D7FF27", "DBFF23", "DFFF1F", "E3FF1B", "E7FF17",
    "EBFF13", "EFFF0F", "F3FF0B", "F7FF07", "FBFF03", "FFFF00", "FFFB00",
    "FFF700", "FFF300", "FFEF00", "FFEB00", "FFE700", "FFE300", "FFDF00",
    "FFDB00", "FFD700", "FFD300", "FFCF00", "FFCB00", "FFC700", "FFC300",
    "FFBF00", "FFBB00", "FFB700", "FFB300", "FFAF00", "FFAB00", "FFA700",
    "FFA300", "FF9F00", "FF9B00", "FF9700", "FF9300", "FF8F00", "FF8B00",
    "FF8700", "FF8300", "FF7F00", "FF7B00", "FF7700", "FF7300", "FF6F00",
    "FF6B00", "FF6700", "FF6300", "FF5F00", "FF5B00", "FF5700", "FF5300",
    "FF4F00", "FF4B00", "FF4700", "FF4300", "FF3F00", "FF3B00", "FF3700",
    "FF3300", "FF2F00", "FF2B00", "FF2700", "FF2300", "FF1F00", "FF1B00",
    "FF1700", "FF1300", "FF0F00", "FF0B00", "FF0700", "FF0300", "FF0000",
    "FB0000", "F70000", "F30000", "EF0000", "EB0000", "E70000", "E30000",
    "DF0000", "DB0000", "D70000", "D30000", "CF0000", "CB0000", "C70000",
    "C30000", "BF0000", "BB0000", "B70000", "B30000", "AF0000", "AB0000",
    "A70000", "A30000", "9F0000", "9B0000", "970000", "930000", "8F0000",
    "8B0000", "870000", "830000", "7F0000"]


class SensorTargetsApp(tk.Frame):
    """ Main app class.
    """

    def __init__(self, master):
        """ Init the GUI components and the Walabot API.
        """
        tk.Frame.__init__(self, master)
        self.canvasPanel = CanvasPanel(self)
        self.canvasHeatmap = CanvasHeatPanel(self)
        self.wbPanel = WalabotPanel(self)
        self.cnfgPanel = ConfigPanel(self)
        self.trgtsPanel = TargetsPanel(self)
        self.ctrlPanel = ControlPanel(self)
        self.canvasPanel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
        self.canvasHeatmap.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
        self.wbPanel.pack(side=tk.TOP, anchor=tk.W, fill=tk.BOTH)
        self.cnfgPanel.pack(side=tk.TOP, anchor=tk.W, fill=tk.BOTH)
        self.trgtsPanel.pack(side=tk.TOP, anchor=tk.W, fill=tk.BOTH)
        self.ctrlPanel.pack(side=tk.TOP, anchor=tk.W, fill=tk.BOTH)
        self.wb = Walabot()

    def initAppLoop(self):
        """ Executed when 'Start' button gets pressed.
            Connect to the Walabot device, set it's arena parameters according
            to one's given by the user, calibrate if needed and start calls
            the loop function.
        """
        self.ctrlPanel.errorVar.set("")
        if self.wb.isConnected():  # connection achieved
            self.ctrlPanel.statusVar.set(self.wb.getStatusString())
            self.update_idletasks()
            try:
                self.wb.setParameters(*self.wbPanel.getParameters())
            except WalabotAPI.WalabotError as err:
                self.ctrlPanel.errorVar.set(str(err))
                return
            params = self.wb.getParameters()
            self.wbPanel.setParameters(*params)  # update entries
            self.canvasPanel.initArenaGrid(*params)  # but only needs R and Phi
            self.numOfTargetsToDisplay = self.cnfgPanel.numTargets.get()
            if not params[4]:  # if not mti
                self.ctrlPanel.statusVar.set(self.wb.getStatusString())
                self.update_idletasks()
                self.wb.calibrate()
            self.lenOfPhi, self.lenOfR = self.wb.getRawImageSliceDimensions()
            self.canvasHeatmap.setGrid(self.lenOfPhi, self.lenOfR)
            self.ctrlPanel.statusVar.set(self.wb.getStatusString())
            self.wbPanel.changeEntriesState("disabled")
            self.cnfgPanel.changeConfigsState("disabled")
            self.ctrlPanel.changeButtonsState("disabled")
            self.loop()
        else:
            self.ctrlPanel.statusVar.set("STATUS_DISCONNECTED")

    def loop(self):
        """ Triggers the Walabot, get the Sensor targets, and update the
            canvas and other components accordingly.
        """
        try:
            rawImage = self.wb.triggerAndGetRawImageSlice()
        except WalabotAPI.WalabotError as err:
            self.ctrlPanel.errorVar.set(str(err))
            self.stopLoop()
            return
        self.canvasHeatmap.update(rawImage, self.lenOfPhi, self.lenOfR)

        try:
            targets = self.wb.getTargets()
        except WalabotAPI.WalabotError as err:
            self.ctrlPanel.errorVar.set(str(err))
            self.stopLoop()
            return
        targets = targets[:self.numOfTargetsToDisplay]
        self.canvasPanel.addTargets(targets)
        self.trgtsPanel.update(targets)
        self.ctrlPanel.fpsVar.set((int(self.wb.getFps())))
        self.cyclesId = self.after_idle(self.loop)

    def stopLoop(self):
        """ Kills the loop function and reset the relevant app components.
        """
        self.after_cancel(self.cyclesId)
        self.wb.stopAndDisconnect()
        self.wbPanel.changeEntriesState("normal")
        self.cnfgPanel.changeConfigsState("normal")
        self.ctrlPanel.changeButtonsState("normal")
        self.canvasPanel.reset()

        self.trgtsPanel.reset()
        self.ctrlPanel.statusVar.set(self.wb.getStatusString())
        self.ctrlPanel.fpsVar.set("")


class WalabotPanel(tk.LabelFrame):
    """ The frame that sets the Walabot settings.
    """

    class WalabotParameter(tk.Frame):
        """ The frame that sets each Walabot parameter line.
        """

        def __init__(self, master, varVal, minVal, maxVal, defaultVal):
            """ Init the Labels (parameter name, min/max value) and entry.
            """
            tk.Frame.__init__(self, master)
            tk.Label(self, text=varVal).pack(side=tk.LEFT, padx=(0, 5), pady=1)
            self.minVal, self.maxVal = minVal, maxVal
            self.var = tk.StringVar()
            self.var.set(defaultVal)
            self.entry = tk.Entry(self, width=7, textvariable=self.var)
            self.entry.pack(side=tk.LEFT)
            self.var.trace("w", lambda a, b, c, var=self.var: self.validate())
            txt = "[{}, {}]".format(minVal, maxVal)
            tk.Label(self, text=txt).pack(side=tk.LEFT, padx=(5, 20), pady=1)

        def validate(self):
            """ Checks that the entered value is a valid number and between
                the min/max values. Change the font color of the value to red
                if False, else to black (normal).
            """
            num = self.var.get()
            try:
                num = float(num)
                if num < self.minVal or num > self.maxVal:
                    self.entry.config(fg="red")
                    return
                self.entry.config(fg="gray1")
            except ValueError:
                self.entry.config(fg="red")
                return

        def get(self):
            """ Returns the entry value as a float.
            """
            return float(self.var.get())

        def set(self, value):
            """ Sets the entry value according to a given one.
            """
            self.var.set(value)

        def changeState(self, state):
            """ Change the entry state according to a given one.
            """
            self.entry.configure(state=state)

    class WalabotParameterMTI(tk.Frame):
        """ The frame that control the Walabot MTI parameter line.
        """
        def __init__(self, master):
            """ Init the MTI line (label, radiobuttons).
            """
            tk.Frame.__init__(self, master)
            tk.Label(self, text="MTI      ").pack(side=tk.LEFT)
            self.mtiVar = tk.IntVar()
            self.mtiVar.set(0)
            self.true = tk.Radiobutton(
                self, text="True", variable=self.mtiVar, value=2)
            self.false = tk.Radiobutton(
                self, text="False", variable=self.mtiVar, value=0)
            self.true.pack(side=tk.LEFT)
            self.false.pack(side=tk.LEFT)

        def get(self):
            """ Returns the value of the pressed radiobutton.
            """
            return self.mtiVar.get()

        def set(self, value):
            """ Sets the pressed radiobutton according to a given value.
            """
            self.mtiVar.set(value)

        def changeState(self, state):
            """ Change the state of the radiobuttons according to a given one.
            """
            self.true.configure(state=state)
            self.false.configure(state=state)

    def __init__(self, master):
        """ Init the parameters lines.
        """
        tk.LabelFrame.__init__(self, master, text="Walabot Settings")
        self.rMin = self.WalabotParameter(self, "R     Min", 1, 1000, 10.0)
        self.rMax = self.WalabotParameter(self, "R     Max", 1, 1000, 100.0)
        self.rRes = self.WalabotParameter(self, "R     Res", 0.1, 10, 2.0)
        self.tMax = self.WalabotParameter(self, "Theta Max", 1, 90, 20.0)
        self.tRes = self.WalabotParameter(self, "Theta Res", 1, 10, 10.0)
        self.pMax = self.WalabotParameter(self, "Phi   Max", 1, 90, 45.0)
        self.pRes = self.WalabotParameter(self, "Phi   Res", 1, 10, 2.0)
        self.thld = self.WalabotParameter(self, "Threshold", 0.1, 100, 15.0)
        self.mti = self.WalabotParameterMTI(self)
        self.parameters = (
            self.rMin, self.rMax, self.rRes, self.tMax, self.tRes,
            self.pMax, self.pRes, self.thld, self.mti)
        for param in self.parameters:
            param.pack(anchor=tk.W)

    def getParameters(self):
        """ Return the values of all the parameters entries/radiobuttons.
        """
        rParams = (self.rMin.get(), self.rMax.get(), self.rRes.get())
        tParams = (-self.tMax.get(), self.tMax.get(), self.tRes.get())
        pParams = (-self.pMax.get(), self.pMax.get(), self.pRes.get())
        thldParam, mtiParam = self.thld.get(), self.mti.get()
        return rParams, tParams, pParams, thldParam, mtiParam

    def setParameters(self, rParams, tParams, pParams, thldParam, mtiParam):
        """ Set the values of all the parameters according to given ones.
        """
        self.rMin.set(rParams[0])
        self.rMax.set(rParams[1])
        self.rRes.set(rParams[2])
        self.tMax.set(tParams[1])
        self.tRes.set(tParams[2])
        self.pMax.set(pParams[1])
        self.pRes.set(pParams[2])
        self.thld.set(thldParam)
        self.mti.set(mtiParam)

    def changeEntriesState(self, state):
        """ Change the state of all the interactive components (entries,
            radiobuttons) according to a given one.
        """
        for param in self.parameters:
            param.changeState(state)


class ConfigPanel(tk.LabelFrame):
    """ The frame that sets the app settings.
    """

    class NumOfTargets(tk.Frame):
        """ The frame that control the number-of-targets line.
        """

        def __init__(self, master):
            """ Init the line, including a label and radiobuttons.
            """
            tk.Frame.__init__(self, master)
            tk.Label(self, text="Targets:").pack(side=tk.LEFT)
            self.maxNum = 4
            self.num = tk.IntVar()
            self.num.set(1)
            self.radios = []
            for i in range(self.maxNum):
                radio = tk.Radiobutton(
                    self, text="{}".format(i+1), variable=self.num, value=i+1)
                radio.pack(side=tk.LEFT)
                self.radios.append(radio)

        def get(self):
            """ Return the value of the pressed radiobutton.
            """
            return self.num.get()

        def set(self, value):
            """ Set the pressed radiobutton according to a given value.
            """
            self.num.set(value)

        def changeButtonsState(self, state):
            """ Change the radiobuttons state according to a given one.
            """
            for radio in self.radios:
                radio.configure(state=state)

    class ArenaDividors(tk.Frame):
        """ The frame that control the number of arena dividors.
        """

        def __init__(self, master):
            """ Init the line, including a label and radiobuttons.
            """
            tk.Frame.__init__(self, master)
            tk.Label(self, text="Arena Dividors:").pack(side=tk.LEFT)
            self.maxNum = 4
            self.num = tk.IntVar()
            self.num.set(2)
            self.radios = []
            for i in range(self.maxNum):
                radio = tk.Radiobutton(
                    self, text="{}".format(2*i+1),
                    variable=self.num, value=i+1
                )
                radio.pack(side=tk.LEFT)
                self.radios.append(radio)

        def get(self):
            """ Return the value of the pressed radiobutton.
            """
            return self.num.get()

        def set(self, value):
            """ Set the pressed radiobutton according to a given value.
            """
            self.num.set(value)

        def changeButtonsState(self, state):
            """ Change the radiobuttons state according to a given one.
            """
            for radio in self.radios:
                radio.configure(state=state)

    def __init__(self, master):
        """ Init the configurations lines.
        """
        tk.LabelFrame.__init__(self, master, text="App Settings")
        self.numTargets = self.NumOfTargets(self)
        self.arenaDividors = self.ArenaDividors(self)
        self.numTargets.pack(anchor=tk.W)
        self.arenaDividors.pack(anchor=tk.W)

    def changeConfigsState(self, state):
        """ Change the state of all interactive components according to a
            given one.
        """
        self.numTargets.changeButtonsState(state)
        self.arenaDividors.changeButtonsState(state)


class ControlPanel(tk.LabelFrame):
    """ The frame that set the control panel.
    """

    def __init__(self, master):
        """ Init the control panel (buttons, status frames).
        """
        tk.LabelFrame.__init__(self, master, text="Control Panel")
        self.buttonsFrame = tk.Frame(self)
        self.runButton, self.stopButton = self.setButtons(self.buttonsFrame)
        self.statusFrame = tk.Frame(self)
        self.statusVar = self.setVar(self.statusFrame, "APP_STATUS", "")
        self.errorFrame = tk.Frame(self)
        self.errorVar = self.setVar(self.errorFrame, "EXCEPTION", "")
        self.fpsFrame = tk.Frame(self)
        self.fpsVar = self.setVar(self.fpsFrame, "FRAME_RATE", "")
        self.buttonsFrame.grid(row=0, column=0, sticky=tk.W)
        self.statusFrame.grid(row=1, columnspan=2, sticky=tk.W)
        self.errorFrame.grid(row=2, columnspan=2, sticky=tk.W)
        self.fpsFrame.grid(row=3, columnspan=2, sticky=tk.W)

    def setButtons(self, frame):
        """ Create the 'start' and 'stop' buttons.
        """
        runButton = tk.Button(frame, text="Start", command=self.start)
        stopButton = tk.Button(frame, text="Stop", command=self.stop)
        runButton.grid(row=0, column=0)
        stopButton.grid(row=0, column=1)
        return runButton, stopButton

    def setVar(self, frame, varText, default):
        """ Create a status label using given parameters.
        """
        strVar = tk.StringVar()
        strVar.set(default)
        tk.Label(frame, text=(varText).ljust(12)).grid(row=0, column=0)
        tk.Label(frame, textvariable=strVar).grid(row=0, column=1)
        return strVar

    def start(self):
        """ Called when the 'start' button gets pressed and init the app loop.
        """
        self.master.initAppLoop()

    def stop(self):
        """ Called when the 'stop' button gets pressed, and stop the app loop.
        """
        if hasattr(self.master, "cyclesId"):
            self.master.stopLoop()

    def changeButtonsState(self, state):
        """ Change the buttons state according to a given one.
        """
        self.runButton.configure(state=state)


class TargetsPanel(tk.LabelFrame):
    """ The frame that shows the targets coordinates.
    """

    def __init__(self, master):
        """ Init the targets frame.
        """
        tk.LabelFrame.__init__(self, master, text="Targets Panel")
        self.targetLabels = []
        for i in range(self.master.cnfgPanel.numTargets.maxNum):
            label = tk.Label(self, text="#{}:".format(i+1))
            label.pack(anchor=tk.W)
            self.targetLabels.append(label)

    def update(self, targets):
        """ update the targets frame according to the given targets.
        """
        for i in range(self.master.numOfTargetsToDisplay):
            if i < len(targets):
                txt = "#{}:   x: {:3.0f}   y: {:3.0f}   z: {:3.0f}".format(
                    i+1,
                    targets[i].xPosCm,
                    targets[i].yPosCm,
                    targets[i].zPosCm)
                self.targetLabels[i].config(text=txt)
            else:
                self.targetLabels[i].config(text="#{}:".format(i+1))

    def reset(self):
        """ Resets the targets frame.
        """
        for i in range(self.master.numOfTargetsToDisplay):
            self.targetLabels[i].config(text="#{}:".format(i+1))


class CanvasPanel(tk.LabelFrame):
    """ The frame the control the arena canvas and displat the targets.
    """

    def __init__(self, master):
        """ Init a black canvas.
        """
        tk.LabelFrame.__init__(self, master, text="Sensor Targets: R / Phi")
        self.canvas = tk.Canvas(
            self, background="light gray",
            width=CANVAS_LENGTH, height=CANVAS_LENGTH, highlightthickness=0)
        self.canvas.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

    def on_resize(self, event):
        wscale = event.width / self.width
        hscale = event.height / self.height
        self.width = event.width
        self.height = event.height
        self.canvas.scale("all", 0, 0, wscale, hscale)

    def initArenaGrid(self, r, theta, phi, threshold, mti):
        """ Draws arena grid (including divisors).
        """
        self.rMin, self.rMax, self.phi = r[0], r[1], phi[1]
        self.drawArenaGrid()
        self.drawArenaDividors()

    def drawArenaGrid(self):
        """ Draw the arena grid using the canvas 'create_arc' function.
        """
        x0 = -self.width * (1/sin(radians(self.phi)) - 1) / 2
        y0 = 0
        x1 = self.width / 2 * (1/sin(radians(self.phi)) + 1)
        y1 = self.height * 2
        startDeg = 90 - self.phi
        extentDeg = self.phi * 2
        self.canvas.create_arc(
            x0, y0, x1, y1,
            start=startDeg, extent=extentDeg, fill="white", width=2)

    def drawArenaDividors(self):
        """ Draw the arena dividors according to the number that was set in
            the config panel.
        """
        x0, y0 = self.width / 2, self.height
        deg = 0
        arenaDividors = self.master.cnfgPanel.arenaDividors.get()
        while deg < self.phi:
            x1 = self.width / 2 * (
                sin(radians(deg))/sin(radians(self.phi)) + 1)
            x2 = self.width / 2 * (
                sin(radians(-deg))/sin(radians(self.phi)) + 1)
            y1 = self.height * (1 - cos(radians(deg)))
            self.canvas.create_line(x0, y0, x1, y1, fill="#AAA", width=1)
            self.canvas.create_line(x0, y0, x2, y1, fill="#AAA", width=1)
            deg += self.phi / arenaDividors

    def addTargets(self, targets):
        """ Draw the given targets on top of the canvas. Remove the older
            targets first.
        """
        self.canvas.delete("target")
        for i, t in enumerate(targets):
            if i < self.master.numOfTargetsToDisplay:
                x = self.width / 2 * (
                    t.yPosCm / (self.rMax * sin(radians(self.phi))) + 1)
                y = self.height * (1 - t.zPosCm / self.rMax)
                self.canvas.create_oval(
                    x-10, y-10, x+10, y+10,
                    fill=COLORS[int(t[3])], tags="target")
                self.canvas.create_text(
                    x, y, text="{}".format(i+1), tags="target")

    def reset(self, *args):
        """ Remove all the canvas components, leaving it black.
        """
        self.canvas.delete("all")

class CanvasHeatPanel(tk.LabelFrame):
    """ This class is designed to control the canvas area of the app.
    """

    def __init__(self, master):
        """ Initialize the label-frame and canvas.
        """
        tk.LabelFrame.__init__(self, master, text='Raw Image Slice: R / Phi')
        self.canvas = tk.Canvas(
            self, width=CANVAS_LENGTH, height=CANVAS_LENGTH)
        self.canvas.pack()
        self.canvas.configure(background='#'+COLORS2[0])

    def setGrid(self, sizeX, sizeY):
        """ Set the canvas components (rectangles), given the size of the axes.
            Arguments:
                sizeX       Number of cells in Phi axis.
                sizeY       Number of cells in R axis.
        """
        recHeight, recWidth = CANVAS_LENGTH/sizeX, CANVAS_LENGTH/sizeY
        self.cells = [[
            self.canvas.create_rectangle(
                recWidth*col, recHeight*row,
                recWidth*(col+1), recHeight*(row+1),
                width=0)
            for col in range(sizeY)] for row in range(sizeX)]

    def update(self, rawImage, lenOfPhi, lenOfR):
        """ Updates the canvas cells colors acorrding to a given rawImage
            matrix and it's dimensions.
            Arguments:
                rawImage    A 2D matrix contains the current rawImage slice.
                lenOfPhi    Number of cells in Phi axis.
                lenOfR      Number of cells in R axis.
        """
        for i in range(lenOfPhi):
            for j in range(lenOfR):
                self.canvas.itemconfigure(
                    self.cells[lenOfPhi-i-1][j],
                    fill='#'+COLORS2[rawImage[i][j]])

    def reset(self):
        """ Deletes all the canvas components (colored rectangles).
        """
        self.canvas.delete('all')

class Walabot:
    """ Control the Walabot using the Walabot API.
    """

    def __init__(self):
        """ Init the Walabot API.
        """
        self.wb = WalabotAPI
        self.wb.Init()
        self.wb.SetSettingsFolder()

    def isConnected(self):
        """ Try to connect the Walabot device. Return True/False accordingly.
        """
        try:
            self.wb.ConnectAny()
        except self.wb.WalabotError as err:
            if err.code == 19:  # "WALABOT_INSTRUMENT_NOT_FOUND"
                return False
            else:
                raise err
        return True

    def getParameters(self):
        """ Get the arena parameters from the Walabot API.
        """
        r = self.wb.GetArenaR()
        theta = self.wb.GetArenaTheta()
        phi = self.wb.GetArenaPhi()
        threshold = self.wb.GetThreshold()
        mti = self.wb.GetDynamicImageFilter()
        return r, theta, phi, threshold, mti

    def setParameters(self, r, theta, phi, threshold, mti):
        """ Set the arena Parameters according given ones.
        """
        self.wb.SetProfile(self.wb.PROF_SENSOR)
        self.wb.SetArenaR(*r)
        self.wb.SetArenaTheta(*theta)
        self.wb.SetArenaPhi(*phi)
        self.wb.SetThreshold(threshold)
        self.wb.SetDynamicImageFilter(mti)
        self.wb.Start()

    def calibrate(self):
        """ Calibrate the Walabot.
        """
        self.wb.StartCalibration()
        while self.wb.GetStatus()[0] == self.wb.STATUS_CALIBRATING:
            self.wb.Trigger()

    def getStatusString(self):
        """ Return the Walabot status as a string.
        """
        status = self.wb.GetStatus()[0]
        if status == 0:
            return "STATUS_DISCONNECTED"
        elif status == 1:
            return "STATUS_CONNECTED"
        elif status == 2:
            return "STATUS_IDLE"
        elif status == 3:
            return "STATUS_SCANNING"
        elif status == 4:
            return "STATUS_CALIBRATING"

    def getTargets(self):
        """ Trigger the Walabot, retrive the targets according to the desired
            tracker given.
        """
        self.wb.Trigger()
        return self.wb.GetSensorTargets()

    def getRawImageSliceDimensions(self):
        """ Returns the dimensions of the rawImage 2D list given from the
            Walabot SDK.
            Returns:
                lenOfPhi    Num of cells in Phi axis.
                lenOfR      Num of cells in Theta axis.
        """
        return self.wb.GetRawImageSlice()[1:3]

    def triggerAndGetRawImageSlice(self):
        """ Returns the rawImage given from the Walabot SDK.
            Returns:
                rawImage    A rawImage list as described in the Walabot docs.
        """
        self.wb.Trigger()
        return self.wb.GetRawImageSlice()[0]

    def getFps(self):
        """ Return the Walabot FPS (internally, from the API).
        """
        return self.wb.GetAdvancedParameter("FrameRate")

    def stopAndDisconnect(self):
        """ Stop and disconnect from the Walabot.
        """
        self.wb.Stop()
        self.wb.Disconnect()


def sensorTargets():
    """ Main app function. Init the main app class, configure the window
        and start the mainloop.
    """
    root = tk.Tk()
    root.title("Walabot - Sensor Targets and Raw Image")
    iconFile = tk.PhotoImage(file="walabot-icon.gif")
    root.tk.call("wm", "iconphoto", root._w, iconFile)  # set app icon
    root.option_add("*Font", "TkFixedFont")
    SensorTargetsApp(root).pack(fill=tk.BOTH, expand=tk.YES)
    root.geometry("+{}+{}".format(APP_X, APP_Y))  # set window location
    root.update()
    root.minsize(width=root.winfo_reqwidth(), height=root.winfo_reqheight())
    root.mainloop()



if __name__ == "__main__":
    sensorTargets()
