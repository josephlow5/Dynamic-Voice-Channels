from random import randint
from . import data

#JK

async def process_changes(member, after):
    name = [
       '好想找人一起玩',
       '有人要++吗😢😢',
       '有人要陪我玩游戏吗',
       '( ´-ω･)▄︻┻┳══━一'
        ]

    guild = after.channel.guild
    new_voice_channel = await guild.create_voice_channel(name[randint(0,name.len - 1)],category=after.channel.category, bitrate=96000)
    await new_voice_channel.edit(sync_permisions=True)

    await member.move_to(new_voice_channel)

    if len(new_voice_channel.voice_states.keys()) == 0:     # 以防进入创建频道channel 然后在创建频道前该user就离开
        await new_voice_channel.delete()
    else:
        data.add_subchannel(new_voice_channel.id)

async def close_subchannel(after):
    await after.channel.delete()

    data.remove_subchannel(after.channel.id)
