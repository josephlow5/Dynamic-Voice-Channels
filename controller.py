import data

#JK

async def process_changes(member, after):
    name = member.display_name + "的小黑屋"
    #permissions dict : everyone -> cannot modify, can enter
    new_voice_channel = await after.guild.create_voice_channel(name,category=after.category, bitrate=256,
                                                               overwrites=permissions)
    data.add_subchannel(new_voice_channel.id)
    #move member to this channel
    
def close_subchannel(after):
    #delete
    #remove subchannel
