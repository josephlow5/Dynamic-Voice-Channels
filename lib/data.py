def cleanup(some_list):
    new_list = []
    for item in some_list:
        new_list.append(item.strip("\n").strip("\r"))
    return new_list

def read_channels():
    channels = []
    try:
        with open("data/channels.txt") as channelsFile:
            channels = channelsFile.readlines()
    except:
        pass
    return cleanup(channels)
def add_channel(channel_id):
    with open("data/channels.txt","a+") as channelsFile:
        channelsFile.write(str(channel_id)+"\n")
def is_monitor_channel(channel_id):
    if str(channel_id) in read_channels():
        return True
    return False

def read_chat_channels():
    channels = []
    try:
        with open("data/chat_channels.txt") as channelsFile:
            channels = channelsFile.readlines()
    except:
        pass
    return cleanup(channels)
def add_chat_channel(channel_id):
    with open("data/chat_channels.txt","a+") as channelsFile:
        channelsFile.write(str(channel_id)+"\n")
def is_monitor_chat_channel(channel_id):
    if str(channel_id) in read_chat_channels():
        return True
    return False


def read_subchannels():
    channels = []
    try:
        with open("data/subchannels.txt") as channelsFile:
            channels = channelsFile.readlines()
    except:
        pass
    return cleanup(channels)
    
def add_subchannel(channel_id):
    with open("data/subchannels.txt","a+") as channelsFile:
        channelsFile.write(str(channel_id)+"\n")
def remove_subchannel(channel_id):
    with open("data/subchannels.txt") as channelsFile:
        raw = channelsFile.read()
    with open("data/subchannels.txt","w") as channelsFile:
        channelsFile.write(raw.replace(str(channel_id)+"\n",""))
def is_monitor_subchannel(channel_id):
    if str(channel_id) in read_subchannels():
        return True
    return False
