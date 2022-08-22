def read_teached():
    teached = {}
    try:
        with open("data/teach.txt",encoding='utf-8-sig') as teachFile:
            raw_teached = teachFile.read()
    except:
        pass
    for teach in raw_teached.split("\n|x|x|\n"):
        if "\n||x|x||\n" in teach:
            received = teach.split("\n||x|x||\n")[0].strip(" ").strip("\n")
            reply = teach.split("\n||x|x||\n")[1].strip(" ").strip("\n")
            if received in teached:
                listedreply = [teached[received]]
                listedreply.append(reply)
                teached[received] = listedreply
            else:
                teached[received] = reply
    return teached
def add_teached(text_receive,text_reply):
    with open("data/teach.txt",'a',encoding='utf-8-sig') as teachFile:
        teachFile.write(text_receive+"\n||x|x||\n"+text_reply+"\n|x|x|\n")

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
