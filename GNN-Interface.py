from tkinter import *
from tkinter import ttk

def Generate(Console) :
    Console['state'] = 'normal'
    Console.insert(INSERT, "Test\n")
    Console['state'] = 'disabled'
    
    return 0

window_root = Tk()
window_root.title = "GNN interface"

window = ttk.Frame(window_root, padding="3 3 12 12")
window.grid(column=0, row=0, sticky=(N,W,E,S))
window_root.columnconfigure(0, weight=1)
window_root.rowconfigure(0, weight=1)

ttk.Label(window, text="Console Window").grid(column=1, row=1)
console = Text(window, width=80, height=39)
console['state'] = 'disabled'
console.grid(column=1, row=2, rowspan=24)

onglets = ttk.Notebook(window, padding="15 3 3 3")
onglets.grid(column=2, row=1, columnspan=2, rowspan=25)

NewModel_Tab = ttk.Frame(onglets, width=50, height=30)
LoadModel_Tab = ttk.Frame(onglets, width=50, height=30)

onglets.add(NewModel_Tab, text="New Model")
onglets.add(LoadModel_Tab, text="Existing Model")

Overfitting = StringVar()
MaxGen = StringVar()

ttk.Label(NewModel_Tab, text="Overfitting Parameter:").pack(pady=(10,0))

Overfitting_CB = ttk.Combobox(NewModel_Tab, textvariable=Overfitting)
Overfitting_CB['values'] = ('3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15')
Overfitting_CB.current(4)
Overfitting_CB.state(["readonly"])
Overfitting_CB.pack()

ttk.Label(NewModel_Tab, text="Max Generation:").pack()

MaxGen_CB = ttk.Combobox(NewModel_Tab, textvariable=MaxGen)
MaxGen_CB['values'] = ('200', '300', '500', '750', '1000', '1500', '2000', '3000')
MaxGen_CB.current(1)
MaxGen_CB.state(["readonly"])
MaxGen_CB.pack()

B_Generate = Button(NewModel_Tab, text="Generate ...", command = lambda: Generate(console))
B_Generate.pack(pady=(100,10))

#ttk.Label(window, text="Select Brain to use:").grid(column=2, row=3, columnspan=2)

#mse_file = open("mse_file.tab", "rb")
#models_file = open("models.brain", "rb")
    
#mse_values = pickle.load(mse_file)
#models = pickle.load(models_file)

#mse_file.close()
#models_file.close()


window_root.mainloop()