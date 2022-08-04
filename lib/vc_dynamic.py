import data

#JK

async def process_changes(member, after):
    name = member.display_name + "的小黑屋"  
    
    new_voice_channel = await after.guild.create_voice_channel(name,category=after.category, bitrate=256)
    await new_voice_channel.edit(sync_permisions=True)

    await member.move_to(new_voice_channel)

    if len(new_voice_channel.voice_states.keys()) == 0:     # 以防进入创建频道channel 然后在创建频道前该user就离开
        await new_voice_channel.delete()
    else:
        data.add_subchannel(new_voice_channel.id)

async def close_subchannel(after):
    await after.channel.delete()

    data.delete_subchannel(after.channel.id)