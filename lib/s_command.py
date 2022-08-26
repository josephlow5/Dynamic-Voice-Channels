#Shuhong
#Add Category
#Create Monitor Channel -> add to data
#Reply slash command

import json
from . import data
from opencc import OpenCC
import discord
from discord import app_commands

with open("data/config.json", encoding="utf-8") as configFile:
    config = json.load(configFile)
developers = config["developers"]

    
def slash_command_init(client):
    @client.tree.command(description="Setup Chatbot and Dynamic Voice Channels in this guild.")
    async def setup(interaction: discord.Interaction):
        reply = await s_command_setup(interaction.guild,client)         #3.1 Slash Command - Setup.
        await interaction.response.send_message(f'{reply}')

    @client.tree.command(description="Teach the bot to response to certain input.")
    @app_commands.describe(
        text_receive='What you expect me response to',
        text_reply='What you expect me response it with',
    )
    async def learn(interaction: discord.Interaction, text_receive: str, text_reply: str):
        reply = await s_command_learn(text_receive,text_reply)          #3.2 Slash Command - Learn.
        await interaction.response.send_message(f'{reply}', ephemeral=True)

    @client.tree.command(description="Get the current teach txt file of the bot.")
    async def get_teach(interaction: discord.Interaction):
        teach = await s_command_get_teach()
        await interaction.response.send_message(
            file=teach,ephemeral=True)                                  #3.3 Slash Command - GetTeach.

    @client.tree.command(description="Make the bot forget everything we told to him (Developers Only).")
    async def obliviate(interaction: discord.Interaction):
        reply = await s_command_obliviate(interaction.user,client)      #3.4 Slash Command - Setup.
        await interaction.response.send_message(f'{reply}')


async def s_command_setup(guild,client):
    DxB_category = await guild.create_category("DxB")
    DxB_chat_channel = await DxB_category.create_text_channel("我是真的AI")
    data.add_chat_channel(DxB_chat_channel.id)
    DxB_main_channel = await DxB_category.create_voice_channel("大家的黑屋",user_limit=1)
    data.add_channel(DxB_main_channel.id)
    reply = "Chatbot and Dynamic Voice Channels are ready to use!"
    return reply

async def s_command_learn(text_receive,text_reply):
    cc = OpenCC('s2tw')
    text_receive = cc.convert(text_receive)
    text_reply = cc.convert(text_reply)
    
    data.add_teached(text_receive,text_reply)
    reply = "I will reply it next time, thank you for teaching me!"
    return reply

async def s_command_get_teach():
    return discord.File('data/teach.txt')

async def s_command_obliviate(user,client):
    if user.id in developers:
        open("data/teach.txt","w").close()
        reply = "I am sorry, who are you? Who am I?"
    else:
        reply = "Sorry, you cannot use this spell."
    return reply
