import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    Wx = np.array(Wx)                
    Wh = np.array(Wh)                 
    b = np.array(b)                   
    h = np.array(initial_hidden_state)  

    for x in input_sequence:
        x = np.array(x)              
        h = np.tanh(Wx @ x + Wh @ h + b)

    final_hidden_state = np.round(h, 4).tolist()
    return final_hidden_state