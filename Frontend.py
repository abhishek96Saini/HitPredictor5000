import tkinter as tk
import tkinter.ttk as ttk


class GUI:

    def __init__(self):
        # creating root
        self.root = tk.Tk()
        self.root.geometry("600x500")
        self.rows = 0

        while self.rows < 2:
            self.root.rowconfigure(self.rows, weight=1)
            self.root.columnconfigure(self.rows, weight=1)
            self.rows += 1

        # creating a frame for storing all widgets
        self.AppFrame = tk.Frame(self.root)
        self.AppFrame.grid(row=0, column=0)

        # Creating label for title
        self.AppTitle = tk.Label(self.AppFrame, text="Hit Predictor", font=("Arial", 30))
        self.AppTitle.grid(row=0, column=0)
        self.AppTitle2 = tk.Label(self.AppFrame, text="TM", font=("Arial", 8))
        self.AppTitle2.grid(row=0, column=1)

        ''' Creating the first tab of the application... this tab is for the songwriter.'''
        # Creating tabs for 2 use cases
        self.notebook = ttk.Notebook(self.AppFrame)
        self.notebook.grid(row=2, column=0, sticky="W", rowspan=100, columnspan=230)

        # Define our first Tab. This tab contains textbox for
        # entering lyrics and button for applying algorithm.
        self.page1 = ttk.Frame(self.root)
        self.notebook.add(self.page1, text='Lyrics Analysis')

        # Label above textbox to tell user to "Enter lyrics Here".
        self.EnterLyricsHereLabel = ttk.Label(self.page1, text="Enter Lyrics here:", padding=5)
        self.EnterLyricsHereLabel.grid(row=1, column=0)

        # Now create textbox in this "Lyrics Analysis" tab. Lyrics are input in this textbox
        self.lyricsTextBox = tk.Text(self.page1, height=20, width=50)
        self.lyricsTextBox.grid(row=2, column=0)  # , rowspan=50, columnspan=50)

        # Button for Applying processing and prediciton to Lyrics.
        self.processButton = ttk.Button(self.page1, text="Apply Processing", width=40)
        self.processButton.grid(row=3, column=0)

        # Label for showcasing the result.
        self.resultLabel = ttk.Label(self.page1, text="\"Result will be shown here inplace of this text\"",
                                     font=("Arial", 20))
        self.resultLabel.grid(row=5, column=0)

        '''  This part of code now declares the page2, use case of music production team.
        '''

        ''' Creating page2 for Music Production company use cases features'''
        self.page2 = ttk.Frame(self.root)
        self.notebook.add(self.page2, text='Music Extras')

        # main loop the root window
        self.root.mainloop()


if __name__ == '__main__':
    MyGui = GUI()
