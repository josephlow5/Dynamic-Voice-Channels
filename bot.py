#======================================================================================#
#                               机器人源码                                             #
#======================================================================================#

import data
bot_token = "MTAwMzU3NTg1MTQ2MTM4MjE5NQ.GGeavO.E7JJSL_r01H4wZa6WalIkeqWrr2f9SyrzL3R9k"
bot_invite = "https://discord.com/api/oauth2/authorize?client_id=1003575851461382195&permissions=8&scope=bot"

import os, sys
directory = os.getcwd()
sys.path.insert(0, directory+"\lib")
import discord

client=discord.Client(intents=discord.Intents.all())

@client.event
async def on_ready():
    print('龙x蓝准备就绪!")

@client.event
async def on_message(message):  #auto evil mean, annoying chatbot
    #example -> 有人知道怎么修理bug吗 -> 怎么不去把你的人生修理好
    #example -> 哈咯 -> 哈咯,废物
    #example -> 我失恋了 -> 很好,继续寻找你的下一个失败吧
          
#Interaction slash command - add monitor channel (shuhong)
#Add Category
#Create Monitor Channel -> add to data
#Reply slask command

@client.event
async def on_voice_state_update(member, before, after): 
    if after.channel is not None:                    
        if data.is_monitor_channel(after.channel.id):
            await controller.process_changes(member, after)
    else:
        if data.is_monitor_subchannel(after.channel.id): 
            if len(after.channel.voice_states.keys()) == 0:
                await controller.close_subchannel(after)
                
    #Yohane
    #If people deafen = kick
    #If people mute > 3 minutes = send warning
    #If people mute > 5 minutes = then kick
    #If people stream > 3 minutes = ask my friend bot to join

client.run(bot_token)

#======================================================================================#
