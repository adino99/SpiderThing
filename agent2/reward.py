import zmq
import numpy as np
import json 
import pandas as pd
import redis
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ZMQ Communication
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://10.8.0.100:5555")

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_data():

    # Get all keys
    all_keys = redis_client.keys('*')

    # Sort keys
    sorted_keys = sorted(all_keys, key=lambda x: x.decode())

    if len(sorted_keys) == 0:
        return []
    # Iterate and print or process sorted keys
    ret = []
    for key in sorted_keys:
        val = redis_client.get(key.decode())
        # print(val)
        try:
            arr_json = json.loads(val)
            # ret.append(float(arr_json['home']))
            ret.append([float(arr_json['decibel']), float(arr_json['ambient']), float(arr_json['neighbor'])])
        except:
            pass
    return ret

# Evaluator Loop
while True:
    # Receive action from Agent 1
    json_data = socket.recv_string()
    data = json.loads(json_data)

    action = data["action"]
    data_sequence = data["data_sequence"]
    df_home = pd.DataFrame(data_sequence, columns=['decibel', 'ambient', 'home'])

    state_local = get_data()
    
    df_local = pd.DataFrame(state_local, columns=['decibel', 'ambient', 'neighbor'])

    # print("Data & Action:", data_sequence, action)
    # print("Home:", df_home.to_numpy())
    # print("Local:", df_local.to_numpy())

    distance, path = fastdtw(df_home.to_numpy(), df_local.to_numpy(), dist=euclidean)
    print("Distance:", distance)
    # Simulate evaluation based on the decibel level
    # Replace this with actual decibel data evaluation
    decibel_sequence = df_home['decibel']
    average_decibel = sum(decibel_sequence) / len(decibel_sequence)
    print("Average:", average_decibel)
    # Evaluate based on the average decibel level
    if action == 1:
        reward = 1 if average_decibel > 60 else -1    
    else:
        reward = 1 if average_decibel <= 60 else -1    

    print("Reward:", reward)
    # Send reward back to Agent 1
    socket.send_pyobj(reward)
