#Creating GUI with tkinter
import tkinter
from tkinter import *
from chatbot_function import *

global emotion_array
emotion_array = []


def send():
    global emotion_array
    
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        sent_msg = [msg]
        pred = sentiment(sent_msg)
        emotion_pred  = class_names[np.argmax(pred)]
        emotion_array.append(emotion_pred)
        ChatLog.insert(END, "Bot: It seems like based on my emotion prediction, you are currently feeling '" + emotion_pred + "'"+'\n\n')
        mental = mental_state(emotion_array)
        ChatLog.insert(END, "While your current mental state is '" + mental + "'"+'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        
 

base = Tk()
base.title("Buddy")
base.geometry("400x600")
base.iconbitmap('buddy.ico')
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap="word")
ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
ChatLog.insert(END, "Buddy: Hello there! My name is Buddy. My job is to assist in identifying your mental health being, how are you feeling today? "+ '\n\n')
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=486) #x=-24, y=-494, height=-14
ChatLog.place(x=6,y=6, height=486, width=370) #x=-394, y=-114, width=-300
EntryBox.place(x=128, y=501, height=90, width=265) 
SendButton.place(x=6, y=501, height=90)

base.mainloop()
