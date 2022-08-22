#Shuhong
#Add Category
#Create Monitor Channel -> add to data
#Reply slash command

from . import data

async def setup(guild,client):
    DxB_category = await guild.create_category("DxB")
    DxB_chat_channel = await DxB_category.create_text_channel("我是真的AI")
    data.add_chat_channel(DxB_chat_channel.id)
    DxB_main_channel = await DxB_category.create_voice_channel("大家的黑屋",user_limit=1)
    data.add_channel(DxB_main_channel.id)
    reply = "DragonXtBlue is fking ready!"
    return reply

async def learn(text_receive,text_reply):
    data.add_teached(text_receive,text_reply)
    reply = "I will reply it next time, thank you for teaching me!"
    return reply

